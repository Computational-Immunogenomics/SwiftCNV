#!/usr/bin/env python3
import os
import logging
import argparse

from .core import run_from_adata



logger = logging.getLogger('SwiftCNV')


def main():
	parser = argparse.ArgumentParser(description='Run SwiftCNV pipeline')
	parser.add_argument('-i', '--input', action='store', dest='h5ad_path', required=True,
						help='Path to input h5ad object')
	parser.add_argument('-X', '--read-X', action='store_true', dest='read_X',
						help='Raw counts are loaded from <input>.X instead of <input>.layers["counts"]')
	parser.add_argument('-o', '--output', action='store', dest='output_dir', required=True,
						help='Path to output directory')
	parser.add_argument('-a', '--gtf-path', action='store', dest='gtf_file', required=True,
						help='Path to gtf gene annotations file for building gene order')
	parser.add_argument('-c', '--cells', action='store', dest='cells_file', default=None,
						help='TSV/CSV with "cell_name", <reference-col> [and optionally <sample-col>]'
							 '. If not defined, <input>.obs will be used')
	parser.add_argument('--reference-col', action='store', dest='reference_col', default='reference',
						help='Column to find reference status (bool) or <reference-vals> (str)')
	parser.add_argument('--reference-vals', action='store', dest='reference_vals', default=None,
						nargs='+', help='Value(s) of reference cells in <reference-col> column')
	parser.add_argument('-s', '--sample-col', action='store', dest='sample_col', default=None,
						help='Column in <cells>/<input>.obs sample IDs for stratification')
	parser.add_argument('--by-sample', action='store_true', dest='by_sample',
						help='Substract the mean of the reference cells for each sample instead of all together')
	parser.add_argument('--exclude-immune', action='store_true', dest='exclude_immune',
						help='Exclude genes names that start with (HLA-|IGH|IGK|IGL)')
	parser.add_argument('-p', '--plot', action='store_true', dest='plot',
						help='Plot final heatmap and heatmaps by sample')
	parser.add_argument('--hmm', action='store_true', dest='hmm',
						help='Perform HMM segmentation of CNV states')
	parser.add_argument('--hmm-by', action='store', dest='hmm_by', default='subcluster',
						choices=['subcluster', 'sample', 'cell'], help='Stratification for HMM segmentation')
	parser.add_argument('--n-clusters', action='store', dest='n_clusters', type=int, default=3,
						help='Number of clusters for performing HMM segmentation analysis')
	parser.add_argument('--cutoff', action='store', dest='cutoff', type=float, default=0.1,
						help='Remove genes whose mean normalized expression across reference cells is below cutoff')
	parser.add_argument('-t', '--threads', action='store', dest='threads', type=int, default=1,
						help='Number of threads to use in parallel processes (segment cells and clustering by samples)')
	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)
	logging.basicConfig(filename=os.path.join(args.output_dir, 'swiftcnv.log'),
						level=logging.INFO, filemode='w', datefmt='%Y-%m-%d %H:%M:%S',
						format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
	logging.captureWarnings(True)

	run_from_adata(
		adata=args.h5ad_path,
		output_dir=args.output_dir,
		gtf_file=args.gtf_file,
		cells_file=args.cells_file,
		reference_col=args.reference_col,
		reference_vals=args.reference_vals,
		read_X=args.read_X,
		sample_col=args.sample_col,
		exclude_immune=args.exclude_immune,
		plot=args.plot,
		run_hmm=args.hmm,
		hmm_by=args.hmm_by,
		n_clusters=args.n_clusters,
		threads=args.threads,
		cutoff=args.cutoff,
		substract_reference_by_sample=args.by_sample,
	)



if __name__ == '__main__':
	main()