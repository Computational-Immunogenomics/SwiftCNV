import os
import logging
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import BayesianGaussianMixture
from joblib import Parallel, delayed

from . import utils



logger = logging.getLogger('SwiftCNV')


class CNVHMM:
	'''
	A 3-state HMM for CNV segmentation (inferCNV i3 implementation)

	States
	------
	0 : Loss
	1 : Neutral (2 copies / diploid)
	2 : Gain
	'''

	def __init__(self, n_states=3, neutral_state=1, transition_prob=1e-15,
				 emission_means=None, emission_sds=None, gene_groups=None):
		'''
		Parameters
		----------
		n_states : int
			Number of HMM states (default=3 for i3).
		neutral_state : int
			Index of the neutral state (default=1 for i3).
		transition_prob : float
			Probability of transitioning to each *other* state at each step.
		'''
		if n_states not in [3, 6]:
			raise ValueError(f'Unsupported n_states "{n_states}", must be 3 or 6')
		self.n_states = n_states
		self.neutral_state = neutral_state

		# Transition matrix setup
		trans = np.full((n_states, n_states), transition_prob)
		np.fill_diagonal(trans, 1.0 - (n_states - 1) * transition_prob)
		self.log_trans = np.log(trans)

		# Initial state probs (heavily favoring neutral state 1)
		pi = np.full(n_states, 0.01)
		pi[neutral_state] = 1.0
		pi /= pi.sum()
		self.log_pi = np.log(pi)

		# Emission parameters tailored for i3 [Loss, Neutral, Gain]
		if emission_means is None:
			mixed_means = {3: np.array([-0.15, 0.0, 0.15]),
						   6: np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3])}
			self.emission_means = mixed_means.get(n_states)
		else:
			self.emission_means = np.array(emission_means)

		if emission_sds is None:
			self.emission_sds = np.full(n_states, 0.08)
		else:
			self.emission_sds = np.array(emission_sds)

		self.gene_groups_bounds = np.r_[0, np.flatnonzero(gene_groups[1:] != gene_groups[:-1]) + 1, len(gene_groups)]


	def _log_emission(self, obs):
		'''Return (T, n_states) log-emission probabilities.'''
		T = len(obs)
		log_e = np.zeros((T, self.n_states))
		for s in range(self.n_states):
			log_e[:, s] = norm.logpdf(obs, loc=self.emission_means[s], scale=self.emission_sds[s])
		return log_e


	def viterbi(self, obs):
		'''
		Run the Viterbi algorithm to find the most likely state sequence.

		Parameters
		----------
		obs : np.ndarray, shape (T,)
			Observed values (e.g. smoothed log2 ratios for one cell).

		Returns
		-------
		np.ndarray, shape (T,)
			Most likely state at each position.
		'''
		T = len(obs)
		log_e = self._log_emission(obs)
		S = self.n_states

		# Viterbi tables
		V = np.full((T, S), -np.inf)
		bp = np.zeros((T, S), dtype=int)

		V[0] = self.log_pi + log_e[0]
		for t in range(1, T):
			trans_scores = V[t - 1, :, np.newaxis] + self.log_trans
			bp[t] = np.argmax(trans_scores, axis=0)
			V[t] = trans_scores[bp[t], np.arange(S)] + log_e[t]

		# Back-trace
		path = np.zeros(T, dtype=int)
		path[-1] = np.argmax(V[-1])
		for t in range(T - 2, -1, -1):
			path[t] = bp[t + 1, path[t + 1]]
		return path


	def fit(self, cnv_matrix, groups=None, threads=1):
		if groups is None:  # Segment by single cell
			if threads <= 1:
				logger.info(f'    HMM Segmentation by cell using single thread')
				return _segment_worker(self, cnv_matrix, self.gene_groups_bounds)
			chunks = np.array_split(cnv_matrix, threads, axis=0)
			logger.info(f'    HMM Segmentation by cell using {threads} threads')
			results = Parallel(n_jobs=threads)(
				delayed(_segment_worker)(self, chunk, self.gene_groups_bounds)
				for chunk in chunks
			)
			return np.vstack(results), None
		else:  # Segment by sample or subcluster (mean over the group)
			uniq_groups = np.sort(np.unique(groups))
			idx_list = [np.where(groups == group)[0] for group in uniq_groups]
			threads = min(threads, len(idx_list))
			if threads <= 1:
				logger.info(f'    HMM Segmentation using single thread for {len(idx_list)} groups')
				results = [_segment_group_worker(self, cnv_matrix[idx, :], self.gene_groups_bounds) for idx in idx_list]
			else:
				logger.info(f'    HMM Segmentation using {threads} threads for {len(idx_list)} groups')
				results = Parallel(n_jobs=threads)(
					delayed(_segment_group_worker)(self, cnv_matrix[idx, :], self.gene_groups_bounds)
					for idx in idx_list
				)
			return np.vstack(results), uniq_groups


def _segment_group_worker(obj, mat, bounds):
	row = np.mean(mat, axis=0)
	if bounds is None:
		return obj.viterbi(row)
	else:
		res = np.empty(row.shape, dtype=int)
		for a, b in zip(bounds[:-1], bounds[1:]):
			res[a:b] = obj.viterbi(row[a:b])
		return res


def _segment_worker(obj, mat, bounds):
	res = np.empty(mat.shape, dtype=int)
	if bounds is None:
		for i in range(mat.shape[0]):
			res[i] = obj.viterbi(mat[i, :])
	else:
		for i in range(mat.shape[0]):
			for a, b in zip(bounds[:-1], bounds[1:]):
				res[i, a:b] = obj.viterbi(mat[i, a:b])
	return res


def filter_states_with_bgm(cnv_states, cnv_matrix, indices=None, neutral_state=1):
	'''
	Post-processes HMM states using a Bayesian Gaussian Mixture Model
	to filter out false-positive low-confidence CNV blocks.
	'''
	n_rows, n_genes = cnv_states.shape
	out_states = cnv_states.copy()

	segment_means = []
	segment_coords = []

	# 1. Extract the coordinates and mean expression for every HMM segment
	for i in range(n_rows):
		states = cnv_states[i]

		# Find the indices where the state changes to define block bounds
		changes = np.where(states[:-1] != states[1:])[0] + 1
		bounds = np.concatenate(([0], changes, [n_genes]))
		for j in range(len(bounds) - 1):
			a, b = bounds[j], bounds[j+1]
			mat_i = i if indices is None else indices[i]
			segment_means.append(np.mean(cnv_matrix[mat_i, a:b]))
			segment_coords.append((i, a, b, states[a]))

	# Reshape to 2D array: n_samples x n_features
	segment_means = np.array(segment_means).reshape(-1, 1)

	# 2. Fit the Bayesian Gaussian Mixture Model
	bgm = BayesianGaussianMixture(n_components=3, max_iter=500, random_state=42,
								  weight_concentration_prior_type='dirichlet_process')
	bgm.fit(segment_means)

	# 3. Identify the 'Neutral' component (the one with its mean closest to 0.0)
	neutral_comp_idx = np.argmin(np.abs(bgm.means_.flatten()))

	# Get the posterior probabilities for all segments
	probs = bgm.predict_proba(segment_means)

	# 4. Filter out false positives
	for i, start, end, original_state in segment_coords:
		if original_state != neutral_state:
			# Check the probability that this altered segment is actually just Neutral noise
			prob_is_neutral = probs[i, neutral_comp_idx]
			if prob_is_neutral > 0.5:
				out_states[i, start:end] = neutral_state  # Revert the block

	return out_states


### Tumor Subclusters

def get_subclusters(cnv_states, n_clusters=3, linkage='ward', groups=None, threads=1):
	'''
	Agglomerative Clustering from CNV score profiles

	Parameters
	----------
	cnv_states : np.ndarray
		Cells x Genes matrix of CNV scores
	n_clusters : int
		Number of subclusters to identify (default=3)
	linkage : str
		Linkage for agglomerative clustering (default='ward')

	Returns
	-------
	np.ndarray
		subcluster labels
	'''

	if groups is None:
		logger.info(f'    Agglomerative Clustering using single thread for whole matrix')
		return _cluster_worker(cnv_states, n_clusters, linkage)

	idx_list = [np.where(groups == group)[0] for group in sorted(np.unique(groups))]
	threads = min(threads, len(idx_list))
	if threads > 1:
		logger.info(f'    Agglomerative Clustering using {threads} threads for {len(idx_list)} groups')
		results = Parallel(n_jobs=threads, prefer='threads')(
			delayed(_cluster_worker)(cnv_states[idx, :], n_clusters, linkage) for idx in idx_list
		)
	else:
		logger.info(f'    Agglomerative Clustering using single thread for {len(idx_list)} groups')
		results = [_cluster_worker(cnv_states[idx, :], n_clusters, linkage) for idx in idx_list]

	subclusters = np.empty(cnv_states.shape[0], dtype=int)
	for idx, res in zip(idx_list, results):
		subclusters[idx] = res
	return subclusters


def _cluster_worker(mat, n_clusters, linkage):
	model = AgglomerativeClustering(n_clusters=min(n_clusters, mat.shape[0]), linkage=linkage)
	if mat.shape[0] <= 1 or n_clusters <= 1:
		return np.zeros(mat.shape[0], dtype=int)
	return model.fit_predict(mat)


### Run Helper

def run_hmm(x, cell_order=None, gene_order=None, output_dir=None, hmm_by='subcluster',
			n_clusters=3, bgm_filter=False, groups=None, plot=True, threads=1):
	if output_dir is not None:
		os.makedirs(output_dir, exist_ok=True)
	
	if cell_order is None or gene_order is None:
		if cell_order is None and gene_order is None and \
		   hasattr(x, 'cell_order') and hasattr(x, 'gene_order'):
			cell_order = x.cell_order
			gene_order = x.gene_order
			cnv_matrix = x.expr
		else:
			raise ValueError('Provide either a processed SwiftCNV object as x or a '
							 'cnv matrix with cell_order and gene_order dataframes')
	else:
		cnv_matrix = x

	### Cell stratification
	ref_cells = cell_order['reference'].to_numpy().astype(bool)
	if hmm_by == 'cell':
		logger.info(f'    HMM segmentation by single cell')
		subclusters = None
		subclusters_df = None
	else:
		subclusters = np.empty(len(ref_cells), dtype=object)
		if hmm_by == 'subcluster':
			logger.info(f'    HMM segmentation by subclusters')
			if groups is None:
				subclusters[ref_cells] = 'ref'
				subclusters[~ref_cells] = get_subclusters(cnv_matrix[~ref_cells, :], n_clusters=n_clusters).astype(str)
			else:
				subclusters[ref_cells] = np.char.add(groups[ref_cells], '_ref')
				obs_res = get_subclusters(cnv_matrix[~ref_cells, :], n_clusters=n_clusters, groups=groups[~ref_cells], threads=threads)
				subclusters[~ref_cells] = np.char.add(np.char.add(groups[~ref_cells], '_'), obs_res.astype(str))
		elif hmm_by == 'sample':
			logger.info(f'    HMM segmentation by sample')
			if groups is None:
				subclusters[ref_cells] = 'ref'
				subclusters[~ref_cells] = '0'
			else:
				subclusters[ref_cells] = np.char.add(groups[ref_cells], '_ref')
				subclusters[~ref_cells] = np.char.add(groups[~ref_cells], '_0')
		else:
			raise ValueError(f'Unrecogized HMM mode "{hmm_by}", hmm_by must be ["cell", "subcluster", "sample"]')

		if groups is None:
			subclusters_df = pd.DataFrame({'cell_name': cell_order['cell_name'], 'subcluster': subclusters})
		else:
			subclusters_df = pd.DataFrame({'cell_name': cell_order['cell_name'], 'subcluster': subclusters, 'sample': groups}, dtype=str)
			subclusters_df['subcluster_number'] = [c.removeprefix(f'{s}_') for c, s in zip(subclusters_df['subcluster'], subclusters_df['sample'])]
			sample_map = subclusters_df[['sample', 'subcluster']].drop_duplicates().set_index('subcluster')['sample'].to_dict()

		if hmm_by == 'subcluster' and output_dir is not None:
			subclusters_df.to_csv(os.path.join(output_dir, 'tumor_subclusters.tsv.gz'), index=False, sep='\t')

	### Cell segmentation
	logger.info(f'    Performing HMM segmentation...')
	arms = gene_order['chr_arm'].to_numpy()
	HMM = CNVHMM(n_states=3, neutral_state=1, gene_groups=arms)
	cnv_states, uniq_subclusters = HMM.fit(cnv_matrix, groups=subclusters, threads=threads)

	## BGM filter for false positives
	if bgm_filter:
		logger.info(f'    Applying Bayesian Gaussian Mixture filter')
		if hmm_by == 'cell':
			indices = None
		else:
			indices = {i: np.where(subclusters_df['subcluster'] == s)[0] for i, s in enumerate(uniq_subclusters)}
		cnv_states = filter_states_with_bgm(cnv_states, cnv_matrix, indices, neutral_state=1)

	if output_dir is not None:
		idx = cell_order['cell_name'].to_numpy() if hmm_by == 'cell' else uniq_subclusters
		cnv_states_df = pd.DataFrame(cnv_states, index=idx, columns=gene_order['gene'])
		cnv_states_df.to_csv(os.path.join(output_dir, 'cnv_states.tsv.gz'), sep='\t')

		### Plotting
		if plot:
			states_filename = os.path.join(output_dir, 'cnv_states.png')
			logger.info(f'    CNV states plotting...')
			cluster_cells = hmm_by == 'cell'
			ref_rows = ref_cells if hmm_by == 'cell' else np.array([i.endswith('ref') for i in idx])
			sample_rows = groups if hmm_by == 'cell' or groups is None else [sample_map.get(s) for s in idx]
			subclusters_rows = idx if hmm_by == 'subcluster' else None
			if groups is not None and hmm_by == 'subcluster':
				subclusters_rows = [c[len(s)+1:] if s.startswith(s) else c for c, s in zip(subclusters_rows, sample_rows)]
			utils.plot_cnv(cnv_states, ref_cells=ref_rows, regions=arms, vmin=0, vmax=2, vcenter=1,
						   output_file=states_filename, cluster_cells=cluster_cells, add_dendrogram=False,
						   header=False, threads=threads, Sample=sample_rows, Subcluster=subclusters_rows)

	return cnv_states, subclusters_df