import os
import re
import sys
import logging
import argparse
from tabnanny import verbose
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy.stats import median_abs_deviation, gaussian_kde
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.colors as mcolors
import scipy.sparse as sp
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from collections import Counter
from sklearn.cluster import DBSCAN
from kneed import KneeLocator


logger = logging.getLogger('SwiftCNV')


class MalignantClassifier:
    def __init__(self, adata, sample_key='sample', cell_type_key='cell_type', 
                 cell_of_origin=None, sample_type_key='sample_type', outdir=None, verbose=True):
        """
        Initializes the classifier class with paths and metadata keys.
        """
        self.adata = adata 
        self.sample_key = sample_key
        self.cell_type_key = cell_type_key
        self.sample_type_key = sample_type_key
        self.outdir = outdir
        self.verbose = verbose

        if self.sample_key not in self.adata.obs:
            raise ValueError(f"{sample_key} not present in adata.obs")
        
        if self.cell_type_key not in self.adata.obs:
            raise ValueError(f"'{cell_type_key}' not present in adata.obs")

        if self.sample_type_key not in self.adata.obs:
            raise ValueError(f"'{sample_type_key}' not present in adata.obs")


        valid_cell_types = set(self.adata.obs[self.cell_type_key].dropna().unique())

        if cell_of_origin is None:
            raise ValueError(
                "cell_of_origin has not been set or not valid. Please set the tumor cell type(s) of origin, e.g: ['Epithelial', 'Glandular']."
                f" Valid values are: {list(self.adata.obs[self.cell_type_key].dropna().unique())}"
            )

        if isinstance(cell_of_origin, str):
            self.cell_of_origin = [item.strip() for item in cell_of_origin.split(',') if item.strip()]
        else:
            self.cell_of_origin = [str(item).strip() for item in cell_of_origin]

        invalid_types = set(self.cell_of_origin) - valid_cell_types
        if invalid_types:
            raise ValueError(
                f"Invalid cell_of_origin provided: {sorted(invalid_types)}. "
                f"Valid values are: {sorted(valid_cell_types)}"
            )

    
    @staticmethod
    def _vectorized_weighted_pearson(X_df, Y_series, W_series):
        """
        Calculates the weighted Pearson correlation for a matrix of cells (X) 
        against a single reference signature (Y) using weights (W).
        """
        X = X_df.values
        Y = Y_series.values
        W = W_series.values
        
        W_sum = np.sum(W)
        if W_sum == 0:
            return np.zeros(X.shape[0])
            
        # Weighted means
        mean_Y = np.sum(Y * W) / W_sum
        mean_X = np.sum(X * W, axis=1) / W_sum
        
        # Centered variables
        diff_Y = Y - mean_Y
        diff_X = X - mean_X[:, np.newaxis] # Broadcast across arms
        
        # Covariances and Variances
        cov_XY = np.sum(W * diff_X * diff_Y, axis=1) / W_sum
        cov_XX = np.sum(W * diff_X**2, axis=1) / W_sum
        cov_YY = np.sum(W * diff_Y**2) / W_sum
        
        # Calculate correlation, handling zero variance (sd = 0 equivalent)
        denom = np.sqrt(cov_XX * cov_YY)
        corr = np.zeros_like(cov_XY)
        
        # Only compute where variance > 0 to avoid division by zero
        valid_mask = denom > 0
        corr[valid_mask] = cov_XY[valid_mask] / denom[valid_mask]
        
        # Re-wrap into a Pandas Series with cell IDs
        return pd.Series(corr, index=X_df.index).fillna(-1)

    @staticmethod
    def _get_clipped_distance(ref_vector, query_matrix, clipped=True):
        """
        Calculates the distance from a reference centroid vector to a query matrix.
        """
        # Convert pandas structures to raw numpy arrays for fast math
        ref_vals = ref_vector.values
        query_vals = query_matrix.values
        
        if clipped:
            dist_matrix = np.sign(ref_vals) * (ref_vals - query_vals)
            
            # Replace any value less than 0 with 0 (equivalent to dist_matrix[dist_matrix < 0] <- 0)
            dist_matrix = np.maximum(dist_matrix, 0)
        else:
            # Simple absolute distance
            dist_matrix = np.abs(ref_vals - query_vals)
            
        # Sum across the arms 
        dist_per_cell = dist_matrix.sum(axis=1)
        
        return pd.Series(dist_per_cell, index=query_matrix.index)

    @staticmethod
    def _get_dynamic_cutoff(scores, strictness, sample_id):
        """
        Calculates the dynamic cutoff by finding KDE peaks and valleys
        """
        # Filter valid scores
        scores = scores[np.isfinite(scores)]
        
        if len(scores) < 2:
            logging.warning(f"Warning: Not enough valid scores (< 2) to compute density in sample ({sample_id}). Returning 0.5")
            return 0.5
            

        # Check if all values are identical (zero variance)
        if np.max(scores) == np.min(scores):
            logging.warning(f"Warning: Scores have zero variance in sample ({sample_id}). Returning 0.5")
            return 0.5


        # Calculate Kernel Density
        try:
            kde = gaussian_kde(scores, bw_method=lambda k: k.scotts_factor() * 1.5)
        except np.linalg.LinAlgError:
            # Catch any remaining linear algebra errors from SciPy
            logging.warning(f"Warning: KDE failed numerically in sample ({sample_id}). Returning 0.5")
            return 0.5
        
        # Create a grid across the data range with padding (simulates R's 512 points)
        padding = np.std(scores)
        x_grid = np.linspace(np.min(scores) - padding, np.max(scores) + padding, 512)
        y_grid = kde(x_grid)
        
        # Find Peaks
        peaks_idx, _ = find_peaks(y_grid)
        if len(peaks_idx) == 0:
            return 0.5
            
        peak_x = x_grid[peaks_idx]
        peak_y = y_grid[peaks_idx]
        
        # Filter tiny noise bumps (at least 5% of max peak)
        min_peak_height = np.max(peak_y) * 0.05
        valid_peaks = peak_y > min_peak_height
        
        peak_x = peak_x[valid_peaks]
        peak_y = peak_y[valid_peaks]
        
        if len(peak_x) < 2:
            logging.warning(f"Warning: could not find two distinct valid peaks in ({sample_id}). Returning 0.5")
            return 0.5
            
        # Identify Normal and Tumor Peaks
        sorted_indices = np.argsort(peak_x)
        peak_x = peak_x[sorted_indices]
        peak_y = peak_y[sorted_indices]
        
        normal_peak = peak_x[0]
        
        if normal_peak > 0.4:
            logging.warning("Warning: Lowest peak is > 0.4 (likely all tumor). Returning default 0.4")
            return 0.4
            
        remaining_x = peak_x[1:]
        remaining_y = peak_y[1:]
        tumor_peak = remaining_x[np.argmax(remaining_y)] # Highest density among the rest
        
        # Find the Valley
        valley_mask = (x_grid > normal_peak) & (x_grid < tumor_peak)
        valley_x_grid = x_grid[valley_mask]
        valley_y_grid = y_grid[valley_mask]
        
        if len(valley_x_grid) == 0:
            return 0.4
            
        valley_x = valley_x_grid[np.argmin(valley_y_grid)]
        
        # Apply Strictness Shift
        if strictness > 0:
            cutoff = valley_x + (tumor_peak - valley_x) * strictness
        else:
            cutoff = valley_x - (valley_x - normal_peak) * abs(strictness)
            
        return cutoff

    @staticmethod
    def _get_dynamic_k(n_cells, sample_id):
        """
        Computes dynamic neighborhood size K based on cell population.
        """
        if n_cells < 10:
            logging.warning(f"Very few reference cells in sample (({sample_id})) Using ({n_cells}) cells as k value in KNN search.")
            return n_cells
        
        # Set K to the square root of N, clamped between 10 and 90
        k_dynamic = int(np.round(np.sqrt(n_cells)))
        return max(10, min(k_dynamic, 90))

    @staticmethod
    def _get_corr_scores_per_sample(sample_id, sample_obs, cnv_mat, cell_of_origin: list, cell_type_key, sample_type_key):
        """Internal helper to calculate scores for a single sample."""
        
        cell_names = sample_obs.index
        
        # Identify Query and Reference cells
        query_cells = cell_names[~sample_obs['reference']]
        normal_cells = cell_names[sample_obs['reference']]

        sample_type = sample_obs[sample_type_key].astype(str).str.lower().unique()[0]

        valid_sample_types = ['tumor', 'normal']

        if sample_type not in valid_sample_types:
            raise ValueError(f'Invalid sample type: {sample_type}. Please set it as "tumor" or "normal".')

        logging.info(f"({sample_id}) Sample type: {sample_type}")
        logging.info(f"({sample_id}) N. infercnv query cells: {len(query_cells)}")
        logging.info(f"({sample_id}) N. infercnv reference cells: {len(normal_cells)}")

        origin_vector = cell_of_origin
        n_malignants = sample_obs[cell_type_key].isin(origin_vector).sum()

        # Check if the sample is healthy (no malignant cells or very few in inferCNV query group)
        if sample_type == 'normal' or len(query_cells) < 20 or n_malignants <= 20:
            logging.info(f"Warning: sample ({sample_id}) does not contain enough query cells (<= 20). Classified as Normal/Unknown.")
            
            chrarms_df = cnv_mat.copy()
            chrarms_df.index.name = 'cell_id'
            chrarms_df = chrarms_df.reset_index()
            chrarms_df = chrarms_df.melt(id_vars=['cell_id'], var_name='chrarms', value_name='cnv_value')
            
            group_map = sample_obs['reference'].map({True: 'Reference', False: 'Query'}).to_dict()
            chrarms_df['group'] = chrarms_df['cell_id'].map(group_map)
            chrarms_df['sample'] = sample_id
            chrarms_df['hotspotarm'] = "No"
            
            corr_df = pd.DataFrame({
                'corr_score': -0.5,
                'corr_state': 'no_corr',
                'sample': sample_id
            }, index=cell_names)
            
            cosine_dist_df = pd.DataFrame({
                'cos_dist': 0.0,
                'sample': sample_id
            }, index=cell_names)
            
            clipped_dist_df = pd.DataFrame({
                'distance_ratio': 0.0,
                'sample': sample_id
            }, index=cell_names)
            
            return {
                'hotspotarms_df': chrarms_df,
                'corr_score': corr_df,
                'cosine_dist': cosine_dist_df,
                'centroids_dist': clipped_dist_df
            }

        # ---------------------------------------------------------
        # Select Hotspot Chromosome Arms
        # ---------------------------------------------------------
        min_mad = 0.005
        
        normal_cnv = cnv_mat.loc[normal_cells]
        malig_cnv = cnv_mat.loc[query_cells]
        
        med_norm = normal_cnv.median(axis=0)
        raw_mad = median_abs_deviation(normal_cnv, axis=0, nan_policy='omit')
        mad_norm = np.maximum(raw_mad, min_mad)

        med_malig = malig_cnv.median(axis=0)

        upper_mad = med_norm + (3 * mad_norm)
        lower_mad = med_norm - (2 * mad_norm)

        gain_mask = med_malig > upper_mad
        loss_mask = med_malig < lower_mad
        hotspotarms = cnv_mat.columns[gain_mask | loss_mask].tolist()

        chrarms_df = cnv_mat.copy()
        chrarms_df.index.name = 'cell_id'
        chrarms_df = chrarms_df.reset_index()

        chrarms_df = chrarms_df.melt(id_vars=['cell_id'], var_name='chrarms', value_name='cnv_value')
        group_map = sample_obs['reference'].map({True: 'Reference', False: 'Query'}).to_dict()
        chrarms_df['group'] = chrarms_df['cell_id'].map(group_map)
        chrarms_df['sample'] = sample_id

        if len(hotspotarms) >= 4:
            logging.info(f"({sample_id}) Nº hotspotarms: {len(hotspotarms)}")

            # remove sexual chromosomes from hotspot arms if present
            sex_arms = ['Xp', 'Xq', 'Yp', 'Yq']
            common = list(set(sex_arms) & set(hotspotarms))

            if common:
                for arm in common:
                    hotspotarms.remove(arm)

            chrarms_df['hotspotarm'] = np.where(chrarms_df['chrarms'].isin(hotspotarms), "Yes", "No")
            cnv_p_mat_sub_hotarms = cnv_mat[hotspotarms]

        else:
            logging.warning(f"({sample_id}) Warning: No hotspot chromosome arms found! Getting hotspot arms from cells of origin only.")
                
            mal_cells = cell_names[sample_obs[cell_type_key].isin(cell_of_origin)]
            
            if len(mal_cells) == 0:
                raise ValueError(f"Error: No cells found matching target cell types: {cell_of_origin}")

            mat_plot_mal = cnv_mat.loc[mal_cells]
            
            # since there is no reference here, hotspot chr arms are chosen based on fraction of cell above a threshold
            median_abs = mat_plot_mal.abs().median(axis=0)
            frac_gain = (mat_plot_mal > 0.05).mean(axis=0)
            frac_loss = (mat_plot_mal < -0.05).mean(axis=0)
            
            fallback_mask = (median_abs > 0.08) & ((frac_gain > 0.5) | (frac_loss > 0.5))
            fallback_hotspots = cnv_mat.columns[fallback_mask].tolist()

            if len(fallback_hotspots) > 0:
                logging.info(f"({sample_id}) Found {len(fallback_hotspots)} hotspot arms from cells of origin!")
                chrarms_df['hotspotarm'] = np.where(chrarms_df['chrarms'].isin(fallback_hotspots), "Yes", "No")
                cnv_p_mat_sub_hotarms = cnv_mat[fallback_hotspots]
            else:
                logging.info(f"({sample_id}) No additional hotspot arms found. Using all arms.")
                chrarms_df['hotspotarm'] = "No"
                cnv_p_mat_sub_hotarms = cnv_mat

        # ---------------------------------------------------------
        # Weighted Correlation Logic
        # ---------------------------------------------------------
        cnv_score_arms = cnv_p_mat_sub_hotarms.abs().sum(axis=1)

        scores_mal = cnv_score_arms.loc[cnv_score_arms.index.isin(query_cells)]
        scores_norm = cnv_score_arms.loc[cnv_score_arms.index.isin(normal_cells)]

        mal_thresh = scores_mal.quantile(0.95)
        norm_thresh = scores_norm.quantile(0.05)

        ref_cells_mal = scores_mal[scores_mal >= mal_thresh].index.tolist()
        ref_cells_norm = scores_norm[scores_norm <= norm_thresh].index.tolist()

        # If there are not enough reference malignant cells, take the top 10
        if len(ref_cells_mal) < 10:
            n_take = min(10, len(scores_mal))
            ref_cells_mal = scores_mal.nlargest(n_take).index.tolist()
        elif len(ref_cells_mal) > 0.8 * len(scores_mal):
            n_take = int(np.ceil(0.05 * len(scores_mal)))
            ref_cells_mal = scores_mal.nlargest(n_take).index.tolist()
            
        if len(ref_cells_norm) < 10:
            n_take = min(10, len(scores_norm))
            ref_cells_norm = scores_norm.nsmallest(n_take).index.tolist()
        elif len(ref_cells_norm) > 0.8 * len(scores_norm):
            n_take = int(np.ceil(0.05 * len(scores_norm)))
            ref_cells_norm = scores_norm.nsmallest(n_take).index.tolist()

        # At least 4 hostpot chr arms required for the correlation
        min_arms_required = 4 
        if len(hotspotarms) >= min_arms_required:
            logging.info(f"({sample_id}) Computing tumor correlation using {len(hotspotarms)} arms.")
            matrix_to_run = cnv_p_mat_sub_hotarms
            ref_signature = matrix_to_run.loc[ref_cells_mal].mean(axis=0)
        else:
            logging.info(f"({sample_id}) Computing correlation on the full matrix.")
            matrix_to_run = cnv_mat
            ref_signature = matrix_to_run.loc[ref_cells_mal].mean(axis=0)

        # Weights are the absolute values of the reference signature
        weights = ref_signature.abs()

        corr_weighted = MalignantClassifier._vectorized_weighted_pearson(matrix_to_run, ref_signature, weights)
        
        dynamic_cut = MalignantClassifier._get_dynamic_cutoff(corr_weighted.values, strictness=0.2, sample_id=sample_id)
        cut_strict = max(0.5, dynamic_cut)

        corr_df = pd.DataFrame({
            'corr_score': corr_weighted.values,
            'corr_cutoff': cut_strict,
            'sample': sample_id
        }, index=corr_weighted.index)

        # -----------------------------------------------
        # Centroids Distance
        # -----------------------------------------------
        ref_signature_malignant = cnv_p_mat_sub_hotarms.loc[ref_cells_mal].mean(axis=0)

        if len(ref_cells_norm) > 10:
            ref_signature_normal = cnv_p_mat_sub_hotarms.loc[ref_cells_norm].mean(axis=0)

        else: #if there are very few normal cells, set a reference signature of 0 in all Chr arms
            logging.warning(f"Warning: Not enough normal reference cells in sample ({sample_id}) (< 10) to calculate a signature. Returning a default vector.")
            # Creates a series of 0s matched to the arm names
            ref_signature_normal = pd.Series(0.0, index=cnv_p_mat_sub_hotarms.columns)

        clipped_dist_mal = MalignantClassifier._get_clipped_distance(
            ref_signature_malignant, 
            cnv_p_mat_sub_hotarms, 
            clipped=True
        )

        clipped_dist_norm = MalignantClassifier._get_clipped_distance(
            ref_signature_normal, 
            cnv_p_mat_sub_hotarms, 
            clipped=False
        )

        distance_ratio = clipped_dist_norm / (clipped_dist_mal + clipped_dist_norm)
        distance_ratio = distance_ratio.fillna(0)

        dyn_cutoff_centroids = MalignantClassifier._get_dynamic_cutoff(
            distance_ratio.values, 
            strictness=0.2, 
            sample_id=sample_id)

        centroids_cutoff = max(0.3, dyn_cutoff_centroids) # minimum cutoff is 0.3

        clipped_dist_df = pd.DataFrame({
            'distance_ratio': distance_ratio,
            'centroids_cutoff': centroids_cutoff,
            'sample': sample_id
        }, index=distance_ratio.index)


        # -----------------------------------------------
        # KNN PCA Cosine Distance
        # -----------------------------------------------
        pca_model = PCA(n_components=None)
        pca_embeddings = pca_model.fit_transform(cnv_mat)

        # Determine dimensions capturing up to 75% variance
        var_per = np.round(pca_model.explained_variance_ratio_ * 100, 1)
        n_dims_pca = max(1, np.sum(np.cumsum(var_per) <= 75))
        pca_embeddings_sub = pca_embeddings[:, :n_dims_pca]

        # Calculate L2 Norms for Cosine mapping conversion
        l2_norms = np.sqrt(np.sum(pca_embeddings_sub**2, axis=1))
        l2_norms[l2_norms == 0] = 1e-10
        pca_mat_l2 = pca_embeddings_sub / l2_norms[:, np.newaxis]

        # Turn L2 matrix into DataFrame to cleanly pull out indexes matching reference cells
        pca_l2_df = pd.DataFrame(pca_mat_l2, index=cnv_mat.index)
        ref_mal_pca = pca_l2_df.loc[ref_cells_mal].values
        ref_norm_pca = pca_l2_df.loc[ref_cells_norm].values

        # KNN Distance to Malignant Profile
        k_mal = MalignantClassifier._get_dynamic_k(ref_mal_pca.shape[0], sample_id=sample_id)
        nn_mal = NearestNeighbors(n_neighbors=k_mal, metric='euclidean').fit(ref_mal_pca)
        dists_mal, _ = nn_mal.kneighbors(pca_mat_l2)
        cos_dist_mal = np.mean((dists_mal**2) / 2, axis=1)

        # KNN Distance to Normal Profile
        if ref_norm_pca.shape[0] > 10:
            k_norm = MalignantClassifier._get_dynamic_k(ref_norm_pca.shape[0], sample_id=sample_id)
            nn_norm = NearestNeighbors(n_neighbors=k_norm, metric='euclidean').fit(ref_norm_pca)
            dists_norm, _ = nn_norm.kneighbors(pca_mat_l2)
            cos_dist_norm = np.mean((dists_norm**2) / 2, axis=1)
        else:
            logging.warning(f"({sample_id}) Warning: Very few normal reference cells (< 10). Returning default distance vector.")
            cos_dist_norm = np.ones(pca_mat_l2.shape[0])

        # Cosine Ratio metric calculation
        knn_cosine_score = cos_dist_norm / (cos_dist_norm + cos_dist_mal)
        knn_cosine_cutoff = max(0.3, MalignantClassifier._get_dynamic_cutoff(knn_cosine_score, strictness=0.1, sample_id=sample_id)) # minimum cutoff is 0.3

        cosine_dist_df = pd.DataFrame({
            'cos_dist': knn_cosine_score,
            'cos_cutoff': knn_cosine_cutoff,
            'sample': sample_id
        }, index=cnv_mat.index)

        return {
            'hotspotarms_df': chrarms_df,
            'corr_score': corr_df,
            'cosine_dist': cosine_dist_df,
            'centroids_dist': clipped_dist_df
        }


    def get_corr_scores(self, n_jobs=-1, obsm_layer='cnv_mat_arms'):
        """
        Calculates scores for all the samples in the adata
        """
        if obsm_layer not in self.adata.obsm:
            raise KeyError(f'{obsm_layer} not present in adata.obsm!')

        unique_samples = self.adata.obs[self.sample_key].unique()
        logging.info(f"Calculating scores across {len(unique_samples)} samples.")

        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1 
        logging.info(f"Using {n_jobs} parallel processes.")
        
        all_hotspots = []
        all_corrs = []
        all_cosines = []
        all_centroids = []
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_sample = {}
            for sample_id in unique_samples:
                mask = self.adata.obs[self.sample_key] == sample_id
                sample_obs = self.adata.obs[mask].copy()
                sample_cnv_mat = self.adata.obsm[obsm_layer].loc[sample_obs.index].copy()
                
                future = executor.submit(
                    MalignantClassifier._get_corr_scores_per_sample,
                    sample_id,
                    sample_obs,
                    sample_cnv_mat,
                    self.cell_of_origin,
                    self.cell_type_key,
                    self.sample_type_key
                )
                future_to_sample[future] = sample_id
            
            for future in as_completed(future_to_sample):
                sample_id = future_to_sample[future]
                try:
                    # Retrieve the dictionary returned by the worker process
                    sample_results = future.result()
                    
                    # Group results into master tracking lists
                    all_hotspots.append(sample_results['hotspotarms_df'])
                    all_corrs.append(sample_results['corr_score'])
                    all_cosines.append(sample_results['cosine_dist'])
                    all_centroids.append(sample_results['centroids_dist'])
                    
                except Exception as e:
                    logging.error(f"Failed scoring on sample ({sample_id}). Error: {str(e)}")
                    raise e
                
        logging.info("Concatenating parallelized sample outputs...")
        self.master_hotspotarms_df = pd.concat(all_hotspots, axis=0, ignore_index=True)
        self.master_corr_df = pd.concat(all_corrs, axis=0)
        self.master_cosine_df = pd.concat(all_cosines, axis=0)
        self.master_centroids_df = pd.concat(all_centroids, axis=0)
        
        logging.info("Successfully executed and aggregated metrics for all samples.")
        

    def plot_cnv_chr_arms_pdf(self, outdir):

        if not hasattr(self, 'master_hotspotarms_df') or self.master_hotspotarms_df is None:
            raise ValueError("Data not found. Run get_corr_scores() first.")

        df = self.master_hotspotarms_df
        sample_ids = sorted(df['sample'].unique())
        
        filename = os.path.join(outdir, "boxplots_cnv_chrArms.pdf")
        
        min_mad = 0.005

        logging.info(f"Generating hotspot arms pdf report for {len(sample_ids)} samples...")

        # Initialize PdfPages to create a multi-page document
        with PdfPages(filename) as pdf:
            
            # Loop through each unique sample to generate its own page
            for sample_id in sample_ids:

                sample_data = df[df['sample'] == sample_id].copy()
                
                # Calculate MAD Thresholds
                normal_data = sample_data[sample_data['group'] == 'Reference']
                mad_records = []
                
                for arm in sample_data['chrarms'].unique():
                    arm_norm = normal_data[normal_data['chrarms'] == arm]['cnv_value']
                    med_norm = arm_norm.median() if len(arm_norm) > 0 else np.nan

                    raw_mad = median_abs_deviation(arm_norm, nan_policy='omit') if len(arm_norm) > 0 else np.nan
                    mad_norm = max(raw_mad, min_mad) if not np.isnan(raw_mad) else np.nan
                    
                    mad_records.append({
                        'chrarms': arm, 
                        'lower_mad': med_norm - (2 * mad_norm), 
                        'upper_mad': med_norm + (3 * mad_norm)
                    })

                mad_thresholds = pd.DataFrame(mad_records)

                # Identify Hotspot Arms
                hotspot_mapping = sample_data[['chrarms', 'hotspotarm']].drop_duplicates()
                hotspot_arms = hotspot_mapping[hotspot_mapping['hotspotarm'] == 'Yes']['chrarms'].tolist()

                unique_arms = sample_data['chrarms'].unique()

                fig, ax = plt.subplots(figsize=(12, 6))

                # Draw Background MAD Thresholds (The grey crossbars)
                arm_to_x = {arm: i for i, arm in enumerate(unique_arms)}

                for _, row in mad_thresholds.iterrows():
                    arm = row['chrarms']
                    if arm in arm_to_x and not np.isnan(row['lower_mad']):
                        x_center = arm_to_x[arm]
                        rect = patches.Rectangle(
                            xy=(x_center - 0.5, row['lower_mad']), 
                            width=1.0, 
                            height=row['upper_mad'] - row['lower_mad'],
                            fill=True, color='grey', alpha=0.5, lw=0, zorder=0
                        )
                        ax.add_patch(rect)

                # Draw the Boxplots
                sns.boxplot(
                    data=sample_data,
                    x='chrarms',
                    y='cnv_value',
                    hue='group',
                    hue_order=['Reference', 'Query'],
                    palette={"Query": "#F8766D", "Reference": "#00BFC4"},
                    showfliers=False, 
                    order=unique_arms,
                    ax=ax,
                    linewidth=1.2,
                    zorder=2
                )

                # Formatting & Theme Customization
                ax.set_ylim(-0.2, 0.2)
                ax.set_xlabel("")
                ax.set_ylabel("cnv_value", fontweight='bold', fontsize=12)
                
                # Update title to dynamically reflect the sample ID
                ax.set_title(f"Sample: {sample_id}", fontsize=16, pad=15) 

                ax.grid(color='grey', alpha=0.2)
                ax.set_axisbelow(True) 
                sns.despine(ax=ax) 

                # Highlight Hotspot Arms (Bold & Red) and Rotate
                for tick_label in ax.get_xticklabels():
                    arm_name = tick_label.get_text()
                    tick_label.set_rotation(90)

                    if arm_name in hotspot_arms:
                        tick_label.set_fontweight('bold')

                # Move Legend to Bottom
                sns.move_legend(
                    ax, "lower center",
                    bbox_to_anchor=(0.5, -0.2), 
                    ncol=2, title=None, frameon=False, fontsize=12
                )

                # Save the current figure to the PDF and close it
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
        logging.info(">> Seaborn hotspot arms report saved!")


    def get_malignant_classif(self, groupby=None):
        """
        Classifies cells into Malignant, Malignant-like, or Normal based on the different metrics using a multi-strictness threshold window.
        """

        if groupby is None:
            groupby = self.sample_key

        corr_df = self.master_corr_df
        centroids_df = self.master_centroids_df
        cosine_df = self.master_cosine_df

        combined_df = pd.concat([
            corr_df[['corr_score', 'corr_cutoff']], 
            centroids_df[['distance_ratio', 'centroids_cutoff']], 
            cosine_df[['cos_dist', 'cos_cutoff', 'sample']]
        ], axis=1)

        aligned_df = combined_df.reindex(self.adata.obs.index)

        # Clean overlap and transfer back to adata.obs
        cols_to_transfer = [
            'corr_score', 'distance_ratio', 'cos_dist', 'corr_cutoff', 'centroids_cutoff', 'cos_cutoff', 'CNV_classif'
            ]
        
        for col in cols_to_transfer:
                if col in aligned_df.columns:
                    self.adata.obs[col] = aligned_df[col]
        
        samples = self.adata.obs['sample'].unique()
        
        for sample in samples:
            adata_sample = self.adata[self.adata.obs['sample'] == sample]

            corr_cutoff = adata_sample.obs['corr_cutoff'].mean()
            cos_cutoff = adata_sample.obs['cos_cutoff'].mean()
            centroid_cutoff = adata_sample.obs['centroids_cutoff'].mean()
            
            corr_state = adata_sample.obs['corr_score'] > corr_cutoff
            cosine_state = adata_sample.obs['cos_dist'] > cos_cutoff
            centroid_state = adata_sample.obs['distance_ratio'] > centroid_cutoff

            conditions = [
                # 1. corr_state == "highly_corr" & centroid_state == "Malignant"
                corr_state & centroid_state,
                
                # 2. corr_state == "highly_corr" & centroids == "Normal" & cosine == "Malignant"
                corr_state & (~centroid_state) & cosine_state,
                
                # 3. corr_state == "highly_corr" & centroids == "Normal" & cosine == "Normal"
                corr_state & (~centroid_state) & (~cosine_state),
                
                # 4. corr_state == "no_corr" & centroids == "Normal"
                (~corr_state) & (~centroid_state),
                
                # 5. corr_state == "no_corr" & centroids == "Malignant" & cosine == "Malignant"
                (~corr_state) & centroid_state & cosine_state,
                
                # 6. corr_state == "no_corr" & centroids == "Malignant" & cosine == "Normal"
                (~corr_state) & centroid_state & (~cosine_state)
            ]

            choices = [
                "Malignant-high confidence",  # 1
                "Malignant-like",             # 2
                "Malignant-like",             # 3
                "Normal",                     # 4
                "Malignant-like",             # 5
                "Normal"                      # 6
            ]

            # Assign classified labels for the sample's indices
            self.adata.obs.loc[adata_sample.obs.index, 'CNV_classif'] = np.select(
                conditions, choices, default="Unknown"
            )

            self.adata.obs['CNV_classif'] = self.adata.obs['CNV_classif'].astype(str)

            for col in self.adata.obs.columns:
                if "arrow" in str(self.adata.obs[col].dtype).lower():
                    self.adata.obs[col] = self.adata.obs[col].astype(object)

        logging.info(">> Successfully computed multi-tier malignant scores and classifications.")


    def get_malignant_score(self):

        def robust_min_max(series, low_q=0.01, high_q=0.99):
            q_low, q_high = series.quantile([low_q, high_q])
            clipped = series.clip(lower=q_low, upper=q_high)

            return (clipped - q_low) / (q_high - q_low)

        samples = self.adata.obs['sample'].unique()
        
        for sample in samples:
            adata_sample = self.adata[self.adata.obs['sample'] == sample]

            mm_corr = robust_min_max(adata_sample.obs['corr_score'])
            mm_cos = robust_min_max(adata_sample.obs['cos_dist'])
            mm_cent = robust_min_max(adata_sample.obs['distance_ratio'])

            weighted_score = (mm_corr * 0.4) + (mm_cos * 0.3) + (mm_cent * 0.3)
            self.adata.obs.loc[adata_sample.obs.index, 'malignant_score'] = weighted_score

 
    def generate_pca(self):
        # check the data type of the matrix
        x_max = self.adata.X.max()
        x_min = self.adata.X.min()

        if x_max > 50 and x_min >= 0:
            logging.info("Matrix values are raw counts")
            data_type = 'counts'

        else:
            if x_min < 0:
                logging.info("Matrix is log scaled")
                data_type = 'scaled log-counts'
            else:
                logging.info("Matrix is log-transformed but not scaled")
                data_type = 'log-counts'
        
        if data_type == "counts":
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)

        # Select HVGs on unscaled log-counts (if not already scaled)
        if data_type != "scaled log-counts":
            sc.pp.highly_variable_genes(self.adata, n_top_genes=2000)
            sc.pp.scale(self.adata, max_value=10)  

       
        use_hvg = "highly_variable" in self.adata.var
        sc.pp.pca(self.adata, mask_var="highly_variable")

        logging.info('X_pca embbeding generated.')
        return self.adata


    def get_majority_vote(self, neighborhood):
        # Filter out missing values or standard string conversions of missing data
        v = [cell for cell in neighborhood if pd.notna(cell) and str(cell).lower() not in ['nan', 'none', 'unknown']]
        
        if not v:
            return "Unknown"
            
        total = len(v)

        malignant_count = sum(1 for cell in v if cell in ["Malignant-high confidence", "Malignant-like"])
            
        # If more than 90% of the cells are Malignant-high confidence, it will be classified as malignant
        if (malignant_count / total) >= 0.90:
            return "Malignant"
        else:
            return "Normal"


    def knn_malignant_classification(self, sample_key, sample_type_key, embedding_key='X_umap'):
        logging.info(">> Computing KNN classification by sample...")

        if embedding_key is None or embedding_key not in self.adata.obsm:
            if 'X_pca' not in self.adata.obsm:
                logging.info('Embbeding key not in adata.obs, generating PCA embbeding.')
                self.generate_pca()
            else:
                logging.info('X_pca found in adata.obs, running knn classification from it.')

            embedding_key = 'X_pca'
            
        self.adata.obs['knn_classif'] = 'Unknown'

        for sample_id in self.adata.obs[sample_key].unique():
            sample_mask = self.adata.obs[sample_key] == sample_id
            
            embeddings_matrix = self.adata.obsm[embedding_key][sample_mask]
            n_cells = embeddings_matrix.shape[0]

            if n_cells > 50:
                k_val = int(np.round(np.sqrt(n_cells)))
                k_val = max(10, min(k_val, 90))

                nn = NearestNeighbors(n_neighbors=k_val, metric='euclidean', n_jobs=-1)
                nn.fit(embeddings_matrix)
                _, nn_indices = nn.kneighbors(embeddings_matrix)

                sample_known_identities = np.asarray(self.adata.obs.loc[sample_mask, 'CNV_classif'])
                nn_identities = sample_known_identities[nn_indices]

                sample_votes = [self.get_majority_vote(row) for row in nn_identities]
                self.adata.obs.loc[sample_mask, 'knn_classif'] = sample_votes

                logging.info(f"KNN completed for sample: {sample_id} using {k_val} neighbours.")

            else:
                logging.warning(f"{sample_id} has less than 50 cells. KNN will not be computed and cells will be classified as their CNV state.")
                CNV_state = self.adata.obs.loc[sample_mask, 'CNV_classif']
                self.adata.obs.loc[sample_mask, 'knn_classif'] = CNV_state

        logging.info(">> KNN classification successfully ran")


    def final_classification(self):
        logging.info(f">> Building final classification.")

        if 'CNV_classif' not in self.adata.obs.columns:
            raise ValueError(f"CNV classification column not found. Run get_malignant_score first.")
            
        if 'knn_classif' not in self.adata.obs.columns:
            raise ValueError(f"KNN classification column not found. Run knn_malignant_classification first.")

        # Pull the data
        cnv_labels = self.adata.obs['CNV_classif'].astype(str).values
        knn_labels = self.adata.obs['knn_classif'].astype(str).values
        
        is_target = self.adata.obs[self.cell_type_key].isin(self.cell_of_origin).values

        conditions = [
            # --- LOGIC FOR TARGET CELLS OF ORIGIN ---
            is_target & (cnv_labels == "Malignant-high confidence"),

            is_target &
            (cnv_labels == "Malignant-like") &
            (knn_labels == "Malignant"),

            is_target &
            (cnv_labels == "Malignant-like") &
            (knn_labels == "Normal"),

            is_target &
            (cnv_labels == "Normal") &
            (knn_labels == "Malignant"),

            is_target &
            (cnv_labels == "Normal") &
            (knn_labels == "Normal"),

            # --- LOGIC FOR ALL OTHER CELLS ---
            ~is_target &
            (cnv_labels == "Malignant-high confidence") &
            (knn_labels == "Malignant"),

            ~is_target &
            (cnv_labels == "Malignant-high confidence") &
            (knn_labels == "Normal"),

            ~is_target &
            (cnv_labels == "Malignant-like") &
            (knn_labels == "Malignant"),

            ~is_target &
            (cnv_labels == "Malignant-like") &
            (knn_labels == "Normal"),

            ~is_target &
            (cnv_labels == "Normal") &
            (knn_labels == "Normal"),
        ]

        choices = [
            # Target cells
            "Malignant-high confidence",
            "Malignant-high confidence",
            "Unknown",
            "Malignant-like",
            "Normal",
            # All other cells
            "Malignant-like",
            "Unknown",
            "Malignant-like",
            "Normal",
            "Normal",
        ]

        # Apply logic and save to adata.obs
        self.adata.obs["malignant_classif"] = np.select(
            conditions,
            choices,
            default="Unknown"
        )
        
        # Convert to category for memory efficiency
        self.adata.obs["malignant_classif"] = self.adata.obs["malignant_classif"].astype("category")
        self.adata.obs["CNV_classif"] = self.adata.obs["CNV_classif"].astype("category")
        self.adata.obs["knn_classif"] = self.adata.obs["knn_classif"].astype("category")



        # remap reference and quary
        self.adata.obs['reference_group'] = (self.adata.obs['reference'].map({True: 'Reference', False: 'Query', 'True': 'Reference', 'False': 'Query'})
        .astype('category'))

        classif_colors = {
            'Malignant-high confidence': '#FA786D',
            'Malignant-like': '#83B701',
            'Normal': '#00C1CA',
            'Unknown': '#C488FF'
            }

        knn_colors = {'Normal': '#00C1CA', 'Malignant':'#FA786D'}

        self.adata.uns['CNV_classif_colors'] = [
            classif_colors[cat] for cat in self.adata.obs['CNV_classif'].cat.categories
        ]


        self.adata.uns['malignant_classif_colors'] = [
            classif_colors[cat] for cat in self.adata.obs['malignant_classif'].cat.categories
        ]

        self.adata.uns['knn_classif_colors'] = [
            knn_colors[cat] for cat in self.adata.obs['knn_classif'].cat.categories
        ]
                

    def dbscan_outlier(self, classif_col='malignant_classif', embedding_key='X_umap', groupby='sample'):
        """
        Runs sample-wise DBSCAN on the UMAP embeddings of malignant cells to detect 
        and label outliers. Creates a True/False flag column.
        """

        logging.info(f">> Running sample-wise DBSCAN outlier detection grouped by '{groupby}'...")
        
        target_classes = ['Malignant-high confidence', 'Malignant-like', 'Malignant'] 
        
        if classif_col not in self.adata.obs:
            raise ValueError(f"Column '{classif_col}' not found in adata.obs.")
            
        malignant_mask = self.adata.obs[classif_col].isin(target_classes)
        all_outlier_indices = []
        
        # Initialize the True/False column with False for all cells
        self.adata.obs['dbscan_outlier'] = False
        
        # Iterate through each sample independently
        for sample_id in self.adata.obs[groupby].unique():
            sample_mask = self.adata.obs[groupby] == sample_id
            
            # Get the exact integer indices in the full dataset for this sample's tumor cells
            subset_indices = np.where(sample_mask & malignant_mask)[0]
            
            if len(subset_indices) > 20:
                db_coords = self.adata.obsm[embedding_key][subset_indices]
                
                # Dynamic k_val tailored to this specific sample's tumor cell count
                k_val = int(np.round(np.sqrt(len(db_coords))))
                k_val = max(10, min(k_val, 90))
                
                # Compute kNN distances
                nn = NearestNeighbors(n_neighbors=k_val)
                nn.fit(db_coords)
                distances, _ = nn.kneighbors(db_coords)
                
                k_distances = distances[:, -1]
                k_distances.sort()
                
                # Find the elbow (eps) for this sample
                x_ranks = np.arange(len(k_distances))
                kneedle = KneeLocator(x_ranks, k_distances, S=1.0, curve='convex', direction='increasing')
                apex = kneedle.elbow_y
                
                if apex is None:
                    apex = np.percentile(k_distances, 90)
                    
                # Run DBSCAN on this sample
                db = DBSCAN(eps=apex, min_samples=k_val)
                clusters = db.fit_predict(db_coords)
                
                # Gather outliers (labeled -1) and map them back to global indices
                outlier_mask = clusters == -1
                outlier_cells_idx = subset_indices[outlier_mask]
                all_outlier_indices.extend(outlier_cells_idx)
                
                logging.info(f"  - {sample_id}: DBSCAN complete (eps: {apex:.4f}, minPts: {k_val}). Found {len(outlier_cells_idx)} outliers.")
                
            elif len(subset_indices) > 0:
                logging.warning(f"  - {sample_id}: Warning - Too few malignant cells ({len(subset_indices)}). Setting all to outliers.")
                all_outlier_indices.extend(subset_indices)

        # Apply the True/False flags and update classifications
        if all_outlier_indices:
            # Set the boolean flag to True for outliers using position-based indexing
            outlier_labels = self.adata.obs.index[all_outlier_indices]
            self.adata.obs.loc[outlier_labels, 'dbscan_outlier'] = True
            
            # Update the categorical classification to 'Unknown'
            updated_classif = self.adata.obs[classif_col].astype(str).values
            updated_classif[all_outlier_indices] = 'Unknown'
            self.adata.obs[classif_col] = updated_classif
            
        self.adata.obs[classif_col] = self.adata.obs[classif_col].astype('category')
        
        logging.info(">> Sample-wise DBSCAN outlier removal done.")

        # print a quick summary
        if self.verbose:
            counts = self.adata.obs["malignant_classif"].value_counts()
            logging.info(">> Final Classification Summary:")
            for status, count in counts.items():
                logging.info(f"   - {status}: {count} cells")

        # format adata properly
        self.adata.obs['CNV_classif'] = self.adata.obs['CNV_classif'].astype('category')
        self.adata.obs['knn_classif'] = self.adata.obs['knn_classif'].astype('category')

        self.adata.var['chr'] = self.adata.var['chr'].astype('category')
        self.adata.var['arm'] = self.adata.var['arm'].astype('category')
        self.adata.var['chr_arm'] = self.adata.var['chr_arm'].astype('category')


    def run_classifiation(self, n_jobs=1, embedding_key='X_umap', report=True):

        self.get_corr_scores(n_jobs=n_jobs)
        self.get_malignant_classif(groupby=self.sample_key)
        self.get_malignant_score()
        self.knn_malignant_classification(self.sample_key, self.sample_type_key, embedding_key= embedding_key)
        self.final_classification()
        self.dbscan_outlier()

        if report:
            self.plot_cnv_chr_arms_pdf(outdir=self.outdir)

        return self.adata

