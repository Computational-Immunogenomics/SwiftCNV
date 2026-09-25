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

from .utils import sort_chrom_arms, get_clusters, merge_clusters


logger = logging.getLogger('SwiftCNV')


class MalignantClassifier:
    def __init__(self, adata, sample_key='sample', cell_type_key='cell_type', 
                 cell_of_origin=None, sample_type_key='sample_type', outdir=None):
        """
        Initializes the classifier class with paths and metadata keys.
        """
        self.adata = adata 
        self.sample_key = sample_key
        self.cell_type_key = cell_type_key
        self.sample_type_key = sample_type_key
        self.outdir = outdir

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
            logger.warning(f"Warning: Not enough valid scores (< 2) to compute density in sample ({sample_id}). Returning 0.5")
            return 0.5
            

        # Check if all values are identical (zero variance)
        if np.max(scores) == np.min(scores):
            logger.warning(f"Warning: Scores have zero variance in sample ({sample_id}). Returning 0.5")
            return 0.5


        # Calculate Kernel Density
        try:
            kde = gaussian_kde(scores, bw_method=lambda k: k.scotts_factor() * 1.5)
        except np.linalg.LinAlgError:
            # Catch any remaining linear algebra errors from SciPy
            logger.warning(f"Warning: KDE failed numerically in sample ({sample_id}). Returning 0.5")
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
            logger.warning(f"Warning: could not find two distinct valid peaks in ({sample_id}). Returning 0.5")
            return 0.5
            
        # Identify Normal and Tumor Peaks
        sorted_indices = np.argsort(peak_x)
        peak_x = peak_x[sorted_indices]
        peak_y = peak_y[sorted_indices]
        
        normal_peak = peak_x[0]
        
        if normal_peak > 0.4:
            logger.warning("Warning: Lowest peak is > 0.4 (likely all tumor). Returning default 0.4")
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
            logger.warning(f"Very few reference cells in sample (({sample_id})) Using ({n_cells}) cells as k value in KNN search.")
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

        logger.info(f"({sample_id}) Sample type: {sample_type}")
        logger.info(f"({sample_id}) N. infercnv query cells: {len(query_cells)}")
        logger.info(f"({sample_id}) N. infercnv reference cells: {len(normal_cells)}")

        origin_vector = cell_of_origin
        n_malignants = sample_obs[cell_type_key].isin(origin_vector).sum()

        # Check if the sample is healthy (no malignant cells or very few in inferCNV query group)
        if sample_type == 'normal' or len(query_cells) < 20 or n_malignants <= 20:
            logger.info(f"Warning: sample ({sample_id}) does not contain enough query cells (<= 20). Classified as Normal/Unknown.")
            
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
            logger.info(f"({sample_id}) Nº hotspotarms: {len(hotspotarms)}")

            # remove sexual chromosomes from hotspot arms if present
            sex_arms = ['Xp', 'Xq', 'Yp', 'Yq']
            common = list(set(sex_arms) & set(hotspotarms))

            if common:
                for arm in common:
                    hotspotarms.remove(arm)

            chrarms_df['hotspotarm'] = np.where(chrarms_df['chrarms'].isin(hotspotarms), "Yes", "No")
            cnv_p_mat_sub_hotarms = cnv_mat[hotspotarms]

        else:
            logger.warning(f"({sample_id}) Warning: No hotspot chromosome arms found! Getting hotspot arms from cells of origin only.")
                
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
                logger.info(f"({sample_id}) Found {len(fallback_hotspots)} hotspot arms from cells of origin!")
                chrarms_df['hotspotarm'] = np.where(chrarms_df['chrarms'].isin(fallback_hotspots), "Yes", "No")
                cnv_p_mat_sub_hotarms = cnv_mat[fallback_hotspots]
            else:
                logger.info(f"({sample_id}) No additional hotspot arms found. Using all arms.")
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
            logger.info(f"({sample_id}) Computing tumor correlation using {len(hotspotarms)} arms.")
            matrix_to_run = cnv_p_mat_sub_hotarms
            ref_signature = matrix_to_run.loc[ref_cells_mal].mean(axis=0)
        else:
            logger.info(f"({sample_id}) Computing correlation on the full matrix.")
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
            logger.warning(f"Warning: Not enough normal reference cells in sample ({sample_id}) (< 10) to calculate a signature. Returning a default vector.")
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
            logger.warning(f"({sample_id}) Warning: Very few normal reference cells (< 10). Returning default distance vector.")
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
        logger.info(f"Calculating scores across {len(unique_samples)} samples.")

        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1 
        logger.info(f"Using {n_jobs} parallel processes.")
        
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
                    logger.error(f"Failed scoring on sample ({sample_id}). Error: {str(e)}")
                    raise e
                
        logger.info("Concatenating parallelized sample outputs...")
        self.master_hotspotarms_df = pd.concat(all_hotspots, axis=0, ignore_index=True)
        self.master_corr_df = pd.concat(all_corrs, axis=0)
        self.master_cosine_df = pd.concat(all_cosines, axis=0)
        self.master_centroids_df = pd.concat(all_centroids, axis=0)
        
        logger.info("Successfully executed and aggregated metrics for all samples.")
        

    def plot_cnv_chr_arms_pdf(self, outdir):

        if not hasattr(self, 'master_hotspotarms_df') or self.master_hotspotarms_df is None:
            raise ValueError("Data not found. Run get_corr_scores() first.")

        df = self.master_hotspotarms_df
        sample_ids = sorted(df['sample'].unique())
        
        filename = os.path.join(outdir, "boxplots_cnv_chrArms.pdf")
        
        min_mad = 0.005

        logger.info(f"Generating hotspot arms pdf report for {len(sample_ids)} samples...")

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
                
        logger.info(">> Seaborn hotspot arms report saved!")


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

        logger.info(">> Successfully computed multi-tier malignant scores and classifications.")


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
            logger.info("Matrix values are raw counts")
            data_type = 'counts'

        else:
            if x_min < 0:
                logger.info("Matrix is log scaled")
                data_type = 'scaled log-counts'
            else:
                logger.info("Matrix is log-transformed but not scaled")
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

        logger.info('X_pca embbeding generated.')
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
        logger.info(">> Computing KNN classification by sample...")

        if embedding_key is None or embedding_key not in self.adata.obsm:
            if 'X_pca' not in self.adata.obsm:
                logger.info('Embbeding key not in adata.obs, generating PCA embbeding.')
                self.generate_pca()
            else:
                logger.info('X_pca found in adata.obs, running knn classification from it.')

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

                logger.info(f"KNN completed for sample: {sample_id} using {k_val} neighbours.")

            else:
                logger.warning(f"{sample_id} has less than 50 cells. KNN will not be computed and cells will be classified as their CNV state.")
                CNV_state = self.adata.obs.loc[sample_mask, 'CNV_classif']
                self.adata.obs.loc[sample_mask, 'knn_classif'] = CNV_state

        logger.info(">> KNN classification successfully ran")


    def final_classification(self):
        logger.info(f">> Building final classification.")

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

        logger.info(f">> Running sample-wise DBSCAN outlier detection grouped by '{groupby}'...")
        
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
                
                logger.info(f"  - {sample_id}: DBSCAN complete (eps: {apex:.4f}, minPts: {k_val}). Found {len(outlier_cells_idx)} outliers.")
                
            elif len(subset_indices) > 0:
                logger.warning(f"  - {sample_id}: Warning - Too few malignant cells ({len(subset_indices)}). Setting all to outliers.")
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
        
        logger.info(">> Sample-wise DBSCAN outlier removal done.")

        # format adata properly
        self.adata.obs['CNV_classif'] = self.adata.obs['CNV_classif'].astype('category')
        self.adata.obs['knn_classif'] = self.adata.obs['knn_classif'].astype('category')

        self.adata.var['chr'] = self.adata.var['chr'].astype('category')
        self.adata.var['arm'] = self.adata.var['arm'].astype('category')
        self.adata.var['chr_arm'] = self.adata.var['chr_arm'].astype('category')


    def plot_cnv_by_sample(self, group_key='sample', cnv_key="cnv_mat_arms", 
        color_by=None, split_by="malignant_classif", continuous_var="malignant_score",
        legend_titles=None, highlight_arms=None, cluster_cells=True, figsize=(20, 12), 
        cmap="RdBu_r", score_cmap="Reds", vmin=None, vmax=None, vcenter=0, threads=-1,
        outdir=None): 

        logging.info(">> Plotting a CNV Heatmap by Sample...")

        if isinstance(color_by, str):
            color_vars = [color_by]
        elif isinstance(color_by, (list, tuple)):
            color_vars = list(color_by)
        else:
            color_vars = []
            
        if legend_titles is None:
            legend_titles = {}
            
        if highlight_arms is None:
            highlight_arms = {}

        # Prevent continuous variables from being treated as categorical
        if continuous_var in color_vars:
            color_vars.remove(continuous_var)

        unique_samples = self.adata.obs[group_key].unique()
        
        # Custom color mappings for categorical variables
        my_colors = {
            'Malignant-high confidence': '#cd5555',
            'Malignant-like': '#ee9572',
            'Normal': '#b2dfee',
            'Unknown': '#b3b3b3'
        }
        knn_colors = {'Normal': '#b2dfee', 'Malignant': '#cd5555'}
        
        # Initialize PDF object if saving

        filename = os.path.join(outdir, "CNV_heatmaps_samples.pdf")

        pdf = PdfPages(filename) if filename else None 
        
        for sample in unique_samples:
            
            # Subset the main anndata object
            sample_adata = self.adata[self.adata.obs[group_key] == sample].copy()
            
            if cnv_key in sample_adata.obsm:
                nested_cnv = sample_adata.obsm[cnv_key]
                if isinstance(nested_cnv, pd.DataFrame):
                    sample_adata.obsm[cnv_key] = nested_cnv.loc[sample_adata.obs_names].copy()
                else:
                    sample_adata.obsm[cnv_key] = nested_cnv[sample_adata.obs_names].copy()
            else:
                raise KeyError(f"'{cnv_key}' not found in adata.obsm for sample '{sample}'.")
    
            # Extract matrix & chromosome metadata
            cnv_adata = sample_adata.obsm[cnv_key]
            mat = cnv_adata.values
            if sp.issparse(mat):
                mat = mat.toarray()

            chromosomes = sort_chrom_arms(cnv_adata.columns)
            n_total_cells = len(sample_adata)

            # Get highlighted arms for this specific sample
            sample_marked_arms = highlight_arms.get(sample, [])

            # Continuous variable setup (CUSTOM HALF-WHITE / HALF-REDS COLORMAP)
            has_continuous = (continuous_var is not None) and (continuous_var in sample_adata.obs.columns)
            if has_continuous:
                score_vals_all = sample_adata.obs[continuous_var].values.astype(float)
                score_vmin = 0.0
                score_vmax = max(1.0, np.nanmax(score_vals_all))
                
                score_norm = mcolors.Normalize(vmin=score_vmin, vmax=score_vmax)
                
                base_cm = plt.colormaps[score_cmap] if isinstance(score_cmap, str) else score_cmap
                n_samples = 256
                half_n = n_samples // 2
                
                white_part = np.tile(np.array([1.0, 1.0, 1.0, 1.0]), (half_n, 1))
                reds_part = base_cm(np.linspace(0.0, 1.0, n_samples - half_n))
                
                score_cm = mcolors.ListedColormap(np.vstack((white_part, reds_part)), name="WhiteToReds")

            # Group and Cluster all cells based on split_by
            cell_groups = {}
            if split_by and split_by in sample_adata.obs.columns:
                raw_vals = sample_adata.obs[split_by].values
                present_vals = [v for v in pd.unique(raw_vals) if pd.notna(v)]
                
                if split_by in ['CNV_classif', 'malignant_classif']:
                    desired_order = ['Normal', 'Malignant-high confidence', 'Malignant-like', 'Unknown']
                    unique_splits = [k for k in desired_order if k in present_vals]
                    unique_splits += [v for v in present_vals if v not in unique_splits]
                elif split_by == 'knn_classif':
                    unique_splits = [k for k in knn_colors.keys() if k in present_vals]
                    unique_splits += [v for v in present_vals if v not in unique_splits]
                else:
                    unique_splits = sorted(present_vals)
                
                for val in unique_splits:
                    group_mask = raw_vals == val
                    group_idx = np.where(group_mask)[0]
                    if len(group_idx) == 0:
                        continue
                    group_mat = mat[group_idx, :]
                    
                    if cluster_cells and len(group_idx) > 1:
                        order, _ = get_clusters(group_mat, threads=threads)
                    else:
                        order = np.arange(len(group_idx))
                        
                    cell_groups[val] = {
                        'mat': group_mat[order, :],
                        'rows': group_idx[order]
                    }
            else:
                all_idx = np.arange(n_total_cells)
                if cluster_cells and len(all_idx) > 1:
                    order, _ = get_clusters(mat, threads=threads)
                else:
                    order = all_idx
                cell_groups['All Cells'] = {
                    'mat': mat[order, :],
                    'rows': all_idx[order]
                }

            # Global Color Palette Setup
            palettes = ["Set3", "tab20", 'Paired'] 
            all_legends_data = []
            global_group_colors = {}

            if color_vars:
                for idx, var in enumerate(color_vars):
                    if var not in sample_adata.obs.columns:
                        continue # Skip missing columns defensively
                        
                    unique_vals = sorted([v for v in pd.unique(sample_adata.obs[var]) if pd.notna(v)])
                    has_nan = sample_adata.obs[var].isna().any()
                    
                    if var in ['CNV_classif', 'malignant_classif']:
                        group_to_color = {g: mcolors.to_rgba(my_colors.get(g, '#CCCCCC')) for g in unique_vals}
                    elif var == 'knn_classif':
                        group_to_color = {g: mcolors.to_rgba(knn_colors.get(g, '#CCCCCC')) for g in unique_vals}
                    else:
                        cat_cmap = plt.colormaps[palettes[idx % len(palettes)]]
                        group_to_color = {g: cat_cmap(i % len(cat_cmap.colors)) for i, g in enumerate(unique_vals)}
                    
                    if has_nan:
                        group_to_color['nan'] = mcolors.to_rgba('#D3D3D3')
                        if 'nan' not in unique_vals:
                            unique_vals.append('nan')
                    
                    global_group_colors[var] = group_to_color
                    handles = [patches.Patch(color=group_to_color[g], label=str(g)) for g in unique_vals]
                    
                    display_title = legend_titles.get(var, var)
                    all_legends_data.append((handles, display_title))

            # Heatmap color scale limits
            if vmin is None or vmax is None:
                p1, p99 = np.percentile(mat.ravel(), [1, 99])
                auto_lim = max(abs(p1), abs(p99))
                auto_lim = max(auto_lim, 0.05)
                vmin, vmax = -auto_lim, auto_lim

            # Build Dynamic GridSpec Layout
            split_gap_height = max(1, int(0.008 * n_total_cells)) 
            chr_height = max(1, int(0.03 * n_total_cells))

            height_ratios = []
            group_keys = list(cell_groups.keys())
            for i, val in enumerate(group_keys):
                height_ratios.append(len(cell_groups[val]['rows']))
                if i < len(group_keys) - 1:
                    height_ratios.append(split_gap_height)
                    
            height_ratios.append(chr_height)
            n_rows = len(height_ratios)

            if has_continuous:
                width_ratios = [5, 0.15, 45, 0.15, 1, 1, 10] 
                col_left_sbar = 0
                col_heatmap = 2
                col_right_sbar = 4
                col_right_panel = 6
            else:
                width_ratios = [5, 0.15, 45, 1, 10]
                col_left_sbar = 0
                col_heatmap = 2
                col_right_sbar = None
                col_right_panel = 4

            fig = plt.figure(figsize=figsize)
            gs = GridSpec(
                n_rows, len(width_ratios), hspace=0.005, wspace=0.005,
                height_ratios=height_ratios, width_ratios=width_ratios
            )

            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            
            # Render Sub-Heatmaps for each classification split
            curr_row = 0
            unique_chroms = np.unique(chromosomes)
            chrom_to_int = {c: i for i, c in enumerate(unique_chroms)}
            chrom_ints = np.array([chrom_to_int[c] for c in chromosomes])

            for i, val in enumerate(group_keys):
                g_data = cell_groups[val]
                n_group_cells = len(g_data['rows'])
                
                # Main CNV Heatmap
                ax_obs = fig.add_subplot(gs[curr_row, col_heatmap])
                im = ax_obs.imshow(g_data['mat'], aspect="auto", cmap=cmap, norm=norm, interpolation="none")
                ax_obs.set_xticks([]); ax_obs.set_yticks([])

                # Categorical Left Sidebars
                if color_vars:
                    gs_obs_sbar = GridSpecFromSubplotSpec(1, len(color_vars), subplot_spec=gs[curr_row, col_left_sbar], wspace=0.15)
                    for c_idx, var in enumerate(color_vars):
                        if var not in global_group_colors: continue # Skip if skipped above
                        
                        ax_obs_var = fig.add_subplot(gs_obs_sbar[0, c_idx])
                        obs_c_mat = np.zeros((n_group_cells, 1, 4))
                        var_vals = sample_adata.obs[var].values[g_data['rows']]
                        
                        for r_idx, v in enumerate(var_vals):
                            lookup_v = 'nan' if pd.isna(v) else v
                            obs_c_mat[r_idx, 0, :] = global_group_colors[var].get(lookup_v, mcolors.to_rgba('#CCCCCC'))
                        
                        ax_obs_var.imshow(obs_c_mat, aspect="auto", interpolation="none")
                        ax_obs_var.set_yticks([])
                        
                        if i == len(group_keys) - 1:
                            ax_obs_var.set_xticks([0])
                            display_title = legend_titles.get(var, var)
                            ax_obs_var.set_xticklabels([display_title], rotation=90, ha="center", va="top", fontsize=9)
                            ax_obs_var.tick_params(axis="x", length=0, pad=2)
                        else:
                            ax_obs_var.set_xticks([])

                        for spine in ax_obs_var.spines.values():
                            spine.set_visible(True); spine.set_color("black"); spine.set_linewidth(1.0)

                # Continuous Right Sidebar
                if has_continuous:
                    ax_score_sbar = fig.add_subplot(gs[curr_row, col_right_sbar])
                    group_scores = sample_adata.obs[continuous_var].values[g_data['rows']].astype(float)
                    
                    score_rgba = np.zeros((n_group_cells, 1, 4))
                    for r_idx, s_val in enumerate(group_scores):
                        if pd.isna(s_val):
                            score_rgba[r_idx, 0, :] = mcolors.to_rgba('#CCCCCC')
                        else:
                            score_rgba[r_idx, 0, :] = score_cm(score_norm(s_val))
                            
                    ax_score_sbar.imshow(score_rgba, aspect="auto", interpolation="none")
                    ax_score_sbar.set_yticks([])
                    ax_score_sbar.set_xticks([])

                    for spine in ax_score_sbar.spines.values():
                        spine.set_visible(True); spine.set_color("black"); spine.set_linewidth(1.0)
                
                # Chromosome boundary lines
                for b in range(1, len(chromosomes)):
                    if chromosomes[b].replace("chr", "")[:-1] != chromosomes[b - 1].replace("chr", "")[:-1]:
                        ax_obs.axvline(b - 0.5, color="#121212", linewidth=1.25, alpha=0.8, zorder=5)
                    else:
                        ax_obs.axvline(b - 0.5, color="#333333", linewidth=1.0, alpha=0.8, zorder=5)
                        
                curr_row += 2

            # Chromosome Track at bottom
            chr_row_idx = n_rows - 1
            ax_chr = fig.add_subplot(gs[chr_row_idx, col_heatmap])
            ax_chr.set_xlim(-0.5, len(chromosomes) - 0.5)
            ax_chr.set_ylim(-0.5, 0.5)
            ax_chr.axis("off")

            for chrom in unique_chroms:
                positions = np.where(chrom_ints == chrom_to_int[chrom])[0]
                mid = positions[len(positions) // 2]
                chrom_label = chrom.replace("chr", "").replace("M", "")
                
                # Check if this arm/chromosome is marked
                is_marked = (chrom in sample_marked_arms or 
                            chrom.replace("chr", "") in sample_marked_arms or 
                            chrom_label in sample_marked_arms)
                
                font_weight = "bold" if is_marked else "normal"
                
                ax_chr.text(
                    mid, 0.3, chrom_label, ha="center", va="top", rotation=90, 
                    fontsize=11, fontweight=font_weight
                )

            # Legends and Colorbars Panel
            heatmap_span_rows = chr_row_idx
            
            gs_right = GridSpecFromSubplotSpec(
                2, 1, 
                subplot_spec=gs[0:heatmap_span_rows, col_right_panel], 
                height_ratios=[1.2, 3.8],
                hspace=0.04
            )
            
            # Colorbars
            if has_continuous:
                gs_cbars_outer = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_right[0, 0], width_ratios=[1.5, 8.5])
                gs_cbars = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_cbars_outer[0, 0], height_ratios=[1, 1], hspace=0.45)
                
                # CNV values Colorbar
                ax_cbar1 = fig.add_subplot(gs_cbars[0, 0])
                fig.colorbar(im, cax=ax_cbar1)
                ax_cbar1.set_title("CNV values", fontsize=11, pad=4, loc="left")
                ax_cbar1.tick_params(labelsize=8)
                ax_cbar1.yaxis.set_ticks_position("right")

                # Continuous Score Colorbar 
                ax_cbar2 = fig.add_subplot(gs_cbars[1, 0])
                sm = plt.cm.ScalarMappable(cmap=score_cm, norm=score_norm)
                sm.set_array([])
                fig.colorbar(sm, cax=ax_cbar2)
                cbar_title = legend_titles.get(continuous_var, continuous_var)
                ax_cbar2.set_title(cbar_title, fontsize=11, pad=4, loc="left")
                ax_cbar2.tick_params(labelsize=8)
                ax_cbar2.yaxis.set_ticks_position("right")
            else:
                gs_cbar = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_right[0, 0], width_ratios=[1.2, 8.8])
                ax_cbar = fig.add_subplot(gs_cbar[0, 0])
                fig.colorbar(im, cax=ax_cbar)
                ax_cbar.set_title("CNV values", fontsize=9, pad=3, loc="left")
                ax_cbar.tick_params(labelsize=8)
                ax_cbar.yaxis.set_ticks_position("right")

            # Categorical Legends
            ax_leg = fig.add_subplot(gs_right[1, 0])
            ax_leg.axis("off")

            leg_y = 1.0 
            for idx, (handles, var_title) in enumerate(all_legends_data):
                leg = ax_leg.legend(
                    handles=handles, title=var_title, loc="upper left", bbox_to_anchor=(0.0, leg_y),
                    ncol=1, fontsize=10, title_fontsize=11, frameon=False,
                    handlelength=1.0, handleheight=1.0, columnspacing=0.8,
                    labelspacing=0.4, borderpad=0.1, handletextpad=0.3, borderaxespad=0.0
                )
                
                leg._legend_box.align = "left" 
                
                ax_leg.add_artist(leg)
                leg_y -= (len(handles) * 0.035) + 0.06

            fig.suptitle(f"Sample: {sample} | {n_total_cells} cells", fontsize=13, y=0.95)
    
            if pdf:
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0.5)
            else:
                plt.show()
                
            plt.close(fig)

        # Close PDF object after the loop finishes
        if pdf:
            pdf.close()

        logging.info(">> CNV Heatmap by Sample succesfully generated!")


    def run_classification(self, n_jobs=1, embedding_key='X_umap', report=True, verbose=True):

        if verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        logger.info(">> Starting malignant classification...")

        self.get_corr_scores(n_jobs=n_jobs)
        self.get_malignant_classif(groupby=self.sample_key)
        self.get_malignant_score()
        self.knn_malignant_classification(self.sample_key, self.sample_type_key, embedding_key= embedding_key)
        self.final_classification()
        self.dbscan_outlier()

        if report:
            self.plot_cnv_chr_arms_pdf(outdir=self.outdir)
            
            # plotting CNV heatmaps
            hotspotarms_dict = (self.master_hotspotarms_df.loc[self.master_hotspotarms_df['hotspotarm'] == 'Yes']
                .groupby('sample')['chrarms']
                .agg(lambda x: x.unique())
                .to_dict()
                )

            titles_dict = {self.cell_type_key: 'Cell type', 'knn_classif': 'KNN classif.', 'CNV_classif': 'CNV classif.', 'CNV_values': 'CNV values', 'malignant_score': 'Malignant score', 'malignant_classif': 'Malignant classif.' }

            self.plot_cnv_by_sample(group_key=self.sample_key, color_by=['malignant_score', self.cell_type_key, 'CNV_classif', 'knn_classif', 'malignant_classif'], split_by='malignant_classif', continuous_var="malignant_score",
                            legend_titles=titles_dict, highlight_arms= hotspotarms_dict, outdir=self.outdir, threads=n_jobs)

        logger.info(">> Malignant classification successfully done!")

        return self.adata

