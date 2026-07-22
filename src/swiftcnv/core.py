import os
import logging

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

from . import hmm
from . import utils



logger = logging.getLogger('SwiftCNV')


class SwiftCNV:
	'''
	Python implementation of the inferCNV algorithm for inferring
	copy number variations from scRNA-seq data.

	Direct implementation of the R inferCNV pipeline:
		https://github.com/broadinstitute/infercnv
	Key reference files in the R package:
	- R/inferCNV.R  (main run method)
	- R/inferCNV_ops.R (individual operations)

	Parameters
	----------
	counts :     scipy.sparse.csc_matrix (cells x genes)
	ref_cells :  np.array
		True for reference cells, False for query (obs) cells
	gene_order : pd.DataFrame
		DataFrame with columns ['gene', 'chr', 'arm', 'chr_arm', 'start', 'end']
	'''

	def __init__(self, counts, cell_order, gene_order):
		if counts.shape[0] == 0 or counts.shape[1] == 0:
			raise ValueError('counts matrix is empty.')
		if counts.shape[0] != len(cell_order):
			raise ValueError(f'Number of cells in counts ({counts.shape[0]}) and cell_order ({len(cell_order)}) don\'t match')
		if counts.shape[1] != len(gene_order):
			raise ValueError(f'Number of genes in counts ({counts.shape[1]}) and gene_order ({len(gene_order)}) don\'t match')

		self.cell_order = cell_order.copy()
		self.gene_order = gene_order.reset_index(drop=True)

		self.is_ref = self.cell_order['reference'].to_numpy()
		self._ref_cells = np.where(self.is_ref)[0]
		self._obs_cells = np.where(~self.is_ref)[0]
		if len(self._ref_cells) == 0:
			raise ValueError(f'No reference cells found!')
		if len(self._obs_cells) == 0:
			raise ValueError(f'No query cells found!')

		if 'sample' in self.cell_order.columns and not np.all(self.cell_order['sample'].isna()):
			self.samples = self.cell_order['sample']
		else:
			self.samples = None

		self.expr = sparse.csr_matrix(counts).astype(np.float64)


	def _filter_and_sort_genes(self, min_cells_per_gene=3):
		'''
		1. Remove genes expressed in fewer than min_cells cells,
		intersect with gene_order, and sort by chromosomal position.

		R equivalent:
			filter_genes_by_count_min_cell, order_by_genome_position
		'''
		n_before = self.expr.shape[1]

		gene_mask = self.expr.getnnz(axis=0) >= min_cells_per_gene
		self.expr = self.expr[:, gene_mask]
		self.gene_order = self.gene_order.loc[gene_mask].reset_index(drop=True)

		self.gene_order = self.gene_order.sort_values(by=['chr', 'start'],
			key=lambda col: col.map(utils.chr_sort_key) if col.name == 'chr' else col)
		self.expr = self.expr[:, self.gene_order.index]
		self.gene_order.reset_index(drop=True, inplace=True)

		n_after = self.expr.shape[1]
		logger.info(f'    1: Gene filtering (min_cells={min_cells_per_gene}):'
					f' from {n_before} to {n_after} genes')


	def _normalize(self):
		'''
		2. Normalize each cell to a fixed library size (CPM-like).

		R equivalent:
			normalize_counts_by_seq_depth
		'''
		cell_totals = self.expr.sum(axis=1)
		scaling_factor = cell_totals.mean()
		if scaling_factor == 0:
			raise ValueError('Mean cell total is 0; cannot normalize.')

		self.expr = self.expr.multiply(scaling_factor / cell_totals).tocsc()

		logger.info(f'    2: Normalization: scaling factor = {scaling_factor:.2f} (mean library size)')


	def _filter_by_cutoff(self, cutoff=0.1):
		'''
		3. Remove genes whose mean normalized expression
		across reference cells is below cutoff

		R equivalent:
			filter_genes_by_cutoff
		'''
		n_before = self.expr.shape[1]
		gene_means = np.asarray(self.expr[self._ref_cells, :].mean(axis=0)).ravel()

		mask = gene_means >= cutoff
		self.expr = self.expr[:, mask]
		self.gene_order = self.gene_order.loc[mask].reset_index(drop=True)

		n_after = self.expr.shape[1]
		logger.info(f'    3: Gene filtering (expression cutoff={cutoff}):'
					f' from {n_before} to {n_after} genes')


	def _log_transform(self):
		'''
		4. Apply log2(x + 1) transform
		*From now on self.expr is a dense np.array

		R equivalent:
			log2xplus1
		'''
		self.expr = np.log2(self.expr.astype(np.float32).toarray() + 1)

		logger.info('    4: Log2(x+1) transform applied')


	def _bound_expression(self, sd_amplifier=3.0):
		'''
		5. Cap extreme values: for each gene,
		clip to [mean - sd*amp, mean + sd*amp]

		R equivalent:
			apply_max_threshold_bounds
		'''
		gene_means = self.expr.mean(axis=0, keepdims=True)
		gene_sds = self.expr.std(axis=0, keepdims=True)

		upper = gene_means + sd_amplifier * gene_sds
		lower = gene_means - sd_amplifier * gene_sds
		self.expr = np.clip(self.expr, lower, upper)

		logger.info(f'    5: Expression bounded at mean +/- {sd_amplifier} x SD per gene')


	def _subtract_reference_mean(self, by_sample=False, _step='6a'):
		'''
		6a/8b. Compute mean expression across reference cells for each gene,
		then subtract from all cells

		R equivalent:
			subtract_ref_expr_from_obs
		'''
		if by_sample and self.samples is not None:
			for sample in np.unique(self.samples):
				mask = self.samples == sample
				ref_idx = self._ref_cells[mask[self._ref_cells]]
				ref_mean = np.mean(self.expr[ref_idx, :], axis=0, keepdims=True)
				self.expr[mask, :] = np.subtract(self.expr[mask, :], ref_mean)
		else:
			ref_mean = np.mean(self.expr[self._ref_cells, :], axis=0, keepdims=True)
			self.expr = np.subtract(self.expr, ref_mean)

		logger.info(f'    {_step}: Reference mean subtracted')


	def _center_cells(self, _step='6b'):
		'''
		6b/8a. Subtract the mean value of each cell across all genes.

		R equivalent:
			center_cell_expr
		'''
		cell_means = self.expr.mean(axis=1, keepdims=True)
		self.expr = np.subtract(self.expr, cell_means)

		logger.info(f'    {_step}: Expression centered by cell')


	def _smooth(self, by='arm', bases_window=3e7, genes_window=75):
		'''
		7. Smooth expression using a window of base pairs and/or genes
		If both are provided, the most restrictive bound is applied

		Parameters
		----------
		by : str ['arm', 'chr']
			Region type to limit smoothing, default is 'arm'
		bases_window : int or None
			Window size in base pairs
		genes_window : int or None
			Window size in genes

		R equivalent:
			smooth_by_chromosome
		'''
		if bases_window is None and genes_window is None:
			raise ValueError('Either "bases_window" or "genes_window" must be not None')
		if by not in ['arm', 'chr']:
			raise ValueError(f'Smoothing can only be done by "arm" or "chr", but "{by}" was given')
		if by == 'arm':
			by = 'chr_arm'

		regions = self.gene_order[by].values

		if bases_window is not None:
			starts = self.gene_order['start'].values
			ends = self.gene_order['end'].values
			centers = (starts + ends) // 2
			bases_half = bases_window // 2

		if genes_window is not None:
			genes_half = genes_window // 2

		for region in np.unique(regions):
			region_idx = np.where(regions == region)[0]
			n = len(region_idx)
			if n == 0:
				continue

			# Bases window bounds
			if bases_window is not None:
				region_centers = centers[region_idx]
				left_bases = np.searchsorted(region_centers, region_centers - bases_half, side='left')
				right_bases = np.searchsorted(region_centers, region_centers + bases_half, side='right')
			else:
				left_bases = np.zeros(n, dtype=int)
				right_bases = np.full(n, n, dtype=int)

			# Gene window bounds
			if genes_window is not None:
				idx = np.arange(n)
				left_genes = np.maximum(0, idx - genes_half)
				right_genes = np.minimum(n, idx + genes_half)
			else:
				left_genes = np.zeros(n, dtype=int)
				right_genes = np.full(n, n, dtype=int)

			# Final bounds
			left = np.maximum(left_bases, left_genes)
			right = np.minimum(right_bases, right_genes)

			# Mean over bounded cumulative sum
			cumsum = np.pad(self.expr[:, region_idx], ((0, 0), (1, 0)), mode='constant').cumsum(axis=1)
			self.expr[:, region_idx] = (cumsum[:, right] - cumsum[:, left]) / (right - left + 1)

		bases_str = f'{bases_window / 1e6:.2f} MB' if bases_window is not None else 'no bases window'
		genes_str = f'{genes_window} genes' if genes_window is not None else 'no genes window'
		logger.info(f'    7: Region smoothing applied ({bases_str}, {genes_str})')


	def _apply_noise_filter(self, noise_filter=0.1, sd_amplifier=1.5, noise_logistic=True):
		'''
		9. Zero out values that are likely noise rather than true CNV signal.
		When noise_logistic=True (default):
			Uses a logistic function centered on the noise threshold to
			smoothly attenuate values near zero
		When noise_logistic=False:
			Hard threshold: if |value| < noise_filter, set to 0

		R equivalent:
			clear_noise_via_ref_mean_sd (if noise_logistic=False)
			clear_noise_via_logistic    (if noise_logistic=True)
		'''
		ref_sd = self.expr[self._ref_cells, :].std()
		threshold = max(noise_filter, ref_sd * sd_amplifier)
		if noise_logistic:
			# Slope controls steepness; R uses a steep logistic
			slope = 50.0  # steep transition
			abs_expr = np.abs(self.expr)
			self.expr /= 1.0 + np.exp(-(np.abs(self.expr) - threshold) * slope)

			logger.info(f'    9: Logistic noise filter applied '
						f'(threshold={threshold:.4f}, ref_sd={ref_sd:.4f})')

		else:
			self.expr[np.abs(self.expr) < threshold] = 0.0

			logger.info(f'    9: Hard noise filter applied '
						f'(threshold={threshold:.4f}, ref_sd={ref_sd:.4f})')


	def _final_bounds(self, cap=1.5):
		'''
		10. Clip the final CNV scores to [-cap, +cap].

		R equivalent:
			apply_max_threshold_bounds
		'''
		self.expr = np.clip(self.expr, -cap, cap)
		logger.info(f'    10: Final values clipped to [-{cap}, +{cap}].')


	def _inverse_log_transform(self):
		'''
		11. Apply 2^x - 1 transform (optional)

		R equivalent:
			invert_log2xplus1
		'''
		self.expr = np.power(2, self.expr) - 1

		logger.info('    11: 2^x - 1 transform applied')



	def run(self, cutoff=0.1, min_cells_per_gene=3, noise_filter=0.1, bound_sd_amplifier=3.0,
			substract_reference_by_sample=False, smooth_by='arm', bases_window=None,
			genes_window=None, denoise=True, noise_logistic=True, sd_amplifier=1.0,
			final_cap=1.5, inv_log=False):
		'''
		Run the full SwiftCNV pipeline.
		'''

		logger.info('=== Starting SwiftCNV pipeline ===')

		# 1. Filter genes & order by chromosome
		self._filter_and_sort_genes(min_cells_per_gene)

		# 2. Normalize (mean library size, matching R)
		self._normalize()

		# 3. Cutoff filter on normalised data, ref cells only (matches R)
		self._filter_by_cutoff(cutoff)

		# 4. Log2 transform
		self._log_transform()

		# 5. Bound extreme values
		self._bound_expression(sd_amplifier=bound_sd_amplifier)

		# 6a. Subtract reference mean
		self._subtract_reference_mean(by_sample=substract_reference_by_sample)

		# 6b. Center each cell before smoothing
		self._center_cells()

		# 7. Smooth
		if bases_window is None and genes_window is None:
			bases_window = 3e7
			genes_window = max(51, round(self.expr.shape[1] / 100))
		self._smooth(by=smooth_by, bases_window=bases_window, genes_window=genes_window)

		# 8a. Re-center each cell after smoothing
		self._center_cells(_step='8a')

		# 8b. Subtract reference median
		self._subtract_reference_mean(by_sample=substract_reference_by_sample, _step='8b')

		# 9. Noise filter
		if denoise:
			self._apply_noise_filter(noise_filter, sd_amplifier, noise_logistic)

		# 10. Final bounding
		self._final_bounds(cap=final_cap)

		# 11. Invert log transform
		if inv_log:
			self._inverse_log_transform()

		logger.info(f'=== SwiftCNV pipeline complete ===')

		return self.expr


	def plot(self, groups=None, region_key='chr_arm', **kwargs):
		if hasattr(self.expr, 'toarray'):
			mat = self.expr.astype(np.float32).toarray()
		else:
			mat = self.expr
		regions = self.gene_order[region_key].to_numpy()
		if groups is None:
			utils.plot_cnv(mat=mat, ref_cells=self.is_ref, regions=regions, **kwargs)
		else:
			if 'output_file' not in kwargs:
				raise ValueError('Need to define output pdf file to save multiplots')
			utils.plot_cnv_multi(mat=mat, ref_cells=self.is_ref, regions=regions, groups=groups, **kwargs)



def run_from_adata(adata, gtf_file, output_dir=None, cells_file=None, reference_col='reference',
				   reference_vals=None, read_X=False, sample_col=None, arms_file=None, exclude_immune=False,
				   plot=False, run_hmm=False, hmm_by='subcluster', n_clusters=3, threads=1, **kwargs):

	n_steps = 4
	if plot and output_dir is not None:
		n_steps += 1
	if run_hmm:
		n_steps += 1
	step = 1


	### 1. Load input data
	if isinstance(adata, ad.AnnData):
		logger.info(f'[{step}/{n_steps}] Using pre-loaded adata...') 
		adata = adata
		preloaded = True
	else:
		logger.info(f'[{step}/{n_steps}] Loading anndata from {adata}...')
		adata = ad.read_h5ad(adata)
		preloaded = False

	genes = adata.var_names.astype(str).to_numpy(dtype=object)
	cell_names = adata.obs_names.astype(str).to_numpy(dtype=object)

	if read_X:
		counts = adata.X
	else:
		counts = adata.layers['counts']
	logger.info(f'    counts shape (cells x genes): {counts.shape}')


	### 2. Get cell order, reference and samples and filter cells
	step += 1
	logger.info(f'[{step}/{n_steps}] Getting reference cells...')
	if cells_file is not None:
		logger.info(f'    Getting reference cells from file: {cells_file}')
		sep = ',' if cells_file.endswith('.csv') or cells_file.endswith('.csv.gz') else '\t'
		counts, cell_order = utils.get_cell_order(cells_file, counts, cell_names, reference_col,
												  reference_vals, sample_col=sample_col, sep=sep)
	elif reference_col in adata.obs.columns:
		if reference_vals is None:
			logger.info(f'    Getting reference cells from adata.obs["{reference_col}"]')
		else:
			logger.info(f'    Getting reference cells from adata.obs["{reference_col}"] (values={reference_vals})')
		counts, cell_order = utils.get_cell_order(adata.obs, counts, cell_names, reference_col,
												  reference_vals, sample_col=sample_col)
	else:
		raise ValueError('Either cells_file or reference_col from adata.obs'
						' (with reference_vals if categories) must be provided')

	ref_cells = cell_order['reference'].to_numpy()
	logger.info(f'    Total cells: {len(cell_order)}')
	logger.info(f'    Reference cells: {np.sum(ref_cells)}')
	logger.info(f'    Observation cells: {np.sum(~ref_cells)}')
	if np.all(ref_cells):
		raise ValueError('No cells were defined as observation')
	elif not np.any(ref_cells):
		raise ValueError('No cells were defined as reference')

	if sample_col is not None:
		sample_ids = cell_order['sample'].to_numpy()
		unique_samples = sorted(np.unique(sample_ids))
		logger.info(f'    Using sample_col: "{sample_col}" with {len(unique_samples)} samples')
	else:
		sample_ids = None
		logger.warning('    No sample_col provided: running as one sample (if you have multiple '
					'samples, use -s/--sample-col for stratification and better performance)')


	### 3. Get gene_order and filter genes
	step += 1
	logger.info(f'[{step}/{n_steps}] Getting gene order...')
	logger.info(f'    Getting gene_order from GTF: {gtf_file}')
	counts, gene_order = utils.get_gene_order(counts, genes, gtf_file, arms_path=arms_file,
											  sex_chr=False, exclude_immune=exclude_immune)
	logger.info(f'    Subsetting to common genes: {len(gene_order)}')


	### 4. Run SwiftCNV
	step += 1
	logger.info(f'[{step}/{n_steps}] Running SwiftCNV...')
	obj = SwiftCNV(counts, cell_order=cell_order, gene_order=gene_order)
	logger.info(f'    SwiftCNV object created: {obj.expr.shape[0]} cells x {obj.expr.shape[1]} genes'
				f' ({len(obj._ref_cells)} reference + {len(obj._obs_cells)} observation cells)')

	cnv_matrix = obj.run(**kwargs)

	# Save CNV matrix and indices
	if output_dir is not None:
		os.makedirs(output_dir, exist_ok=True)
		np.savez_compressed(os.path.join(output_dir, 'cnv_scores.npz'), arr=cnv_matrix)
		obj.cell_order.to_csv(os.path.join(output_dir, 'cell_order.tsv.gz'), sep='\t', index=False)
		obj.gene_order.to_csv(os.path.join(output_dir, 'gene_order.tsv.gz'), sep='\t', index=False)
		logger.info(f'    CNV matrix saved: {cnv_matrix.shape[0]} cells x {cnv_matrix.shape[1]} genes')


	### 5. HMM Segmentation
	if run_hmm:
		step += 1
		logger.info(f'[{step}/{n_steps}] HMM segmentation...')
		hmm_dir = None if output_dir is None else os.path.join(output_dir, 'hmm')
		cnv_states, subclusters_df = hmm.run_hmm(
			obj, output_dir=hmm_dir, groups=sample_ids, hmm_by=hmm_by,
			n_clusters=n_clusters, plot=plot, threads=threads,
		)


	### 6. Plot CNV matrix
	if output_dir is not None and plot:
		step += 1
		logger.info(f'[{step}/{n_steps}] Plotting heatmap...')
		heatmap_filename = os.path.join(output_dir, 'cnv_scores.png')
		subcl = subclusters_df['subcluster_number'] if run_hmm and hmm_by == 'subcluster' else None
		obj.plot(output_file=heatmap_filename, threads=threads, Sample=sample_ids, Subcluster=subcl)
		logger.info(f'    Heatmap saved to {heatmap_filename}')
		if sample_ids is not None:
			p1, p99 = np.percentile(cnv_matrix, [1, 99])
			vmax = max(max(abs(p1), abs(p99)), 0.05)
			vmin = -vmax
			pdf_filename = os.path.join(output_dir, 'cnv_scores_by_sample.pdf')
			obj.plot(output_file=pdf_filename, threads=threads, groups=sample_ids, vmin=vmin, vmax=vmax, Subcluster=subcl)
			logger.info(f'    PDF by sample saved to {pdf_filename}')

	logger.info('Finished')

	if preloaded:
		adata = utils.add_mat_to_adata(adata, cnv_matrix, obj.cell_order, obj.gene_order)
		if run_hmm:
			return adata, cnv_states, subclusters_df
		else:
			return adata
	else:
		if run_hmm:
			return cnv_matrix, cnv_states, subclusters_df
		else:
			return cnv_matrix
