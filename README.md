# SwiftCNV

SwiftCNV is a fast and scalable Python implementation of the core [InferCNV](https://github.com/broadinstitute/inferCNV/wiki) algorithm to infer copy number variations (CNVs) from single-cell RNA-seq data. It provides additional features and is designed for seamless interoperability with Anndata objects and Scanpy.

## Installation

SwiftCNV can be installed through pip:

```bash
pip install swiftcnv
```

Python dependencies are: `numpy`, `pandas`, `scipy`, `scikit-learn`, `anndata`, `matplotlib`

Extra dependencies for the tutorial: `scanpy`, `ipykernel`, `leidenalg`, `requests`

```bash
pip install swiftcnv[tutorial]
```

## Usage

SwiftCNV can be run from the command line from a h5ad file, but can also be imported to your script for advanced usage and AnnData/Scanpy integration. SwiftCNV requires a portion of the cells to be defined as reference to calculate the CNV score, normally being cells that are not expected to be malignant (e.g., based on cell type). The program requires a GTF file containing gene annotations, which be obtained from the [Gencode](https://www.gencodegenes.org/human/) database.

### Command Line Interface

#### Required arguments

| Option | Description |
|--------|-------------|
| `-i`, `--input` | Path to input h5ad object |
| `-o`, `--output` | Path to output directory |
| `-a`, `--gtf-path` | Path to gtf gene annotations file for building gene order |

#### Optional arguments

| Option | Description |
|--------|-------------|
| `-X`, `--read-X` | Raw counts are loaded from `<input>.X`. Default: laoded from `<input>.layers["counts"]` |
| `-c`, `--cells` | TSV with `"cell_name"`, `<reference-col>` (and optionally `<sample-col>`). Default: `<input>.obs` will be used |
| `--reference-col` | Column to find reference status. Default: `reference` |
| `--reference-vals` | Value(s) of reference cells in `<reference-col>` column. Default: `<reference-col>` will be interpreted as bool |
| `-s`, `--sample-col` | Column in `<cells>`/`<input>.obs` sample IDs for stratification. Default: no samples |
| `--by-sample` | Substract the mean of the reference cells for each sample instead of all samples together |
| `--exclude-immune` | Exclude genes names that start with `(HLA-\|IGH\|IGK\|IGL)` to avoid bias from reference immune cells |
| `-p`, `--plot` | Plot final heatmap and heatmaps by sample (if provided) |
| `--hmm` | Perform HMM segmentation of CNV states |
| `--hmm-by` | Stratification for HMM segmentation (`subcluster`, `sample` or `cell`). Default: `subcluster` |
| `--n-clusters` | Number of clusters for performing HMM segmentation analysis. Default: `3` |
| `--cutoff` | Remove genes whose mean normalized expression across reference cells is below cutoff. Default: `0.1` |
| `-t`, `--threads` | Number of threads to use in parallel processes (HMM segmentation and clustering) |

Reference cells can be specified using a TSV file with two columns, cell_name and reference, where the reference column contains TRUE or FALSE to indicate whether each cell is used as a reference. Alternatively, reference cells can be specified by providing the column in adata.obs containing the cell type annotations (--reference-col) and which ones should be used as reference (--reference-value).  Finally, the column identifying the samples must be specified.

#### Example

A typical call from a cells file would be:

```bash
swiftcnv \
    -i /path/to/adata.h5ad \
    -o /path/to/output \
    -a /path/to/gene_annotations.gtf.gz \
    -c /path/to/cells.tsv \
    -s sample \
    -p \
    --hmm
```

Where the required cells.tsv file (`-c` / `--cells`) would be:

| cell_name | reference | sample |
| --- | --- | --- |
| AAATGCCTCACATACG | True | s1 |
| AACCATGGTTATTCTC | False | s1 |
| AACGTTGGTTTACTCT | True | s2 |
| ... | ... | ... |

Alternatively, one can use cell types from the input file to define the reference

```bash
swiftcnv \
    -i /path/to/adata.h5ad \
    -o /path/to/output \
    -a /path/to/gene_annotations.gtf.gz \
    --reference-col cell_type \
    --reference-vals T_cell Macrophage Fibroblast \
    -s sample \
    -p \
    --hmm
```

### Using SwiftCNV in a script or notebook

```python
import logging
import swiftcnv as cnv
from swiftcnv.data import Qian2020_Ovarian

# Enable INFO level logging for verbosity
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')

# Load example dataset
adata = Qian2020_Ovarian()
```

For running manually, first create a SwiftCNV object with the input counts matrix (will be converted to scipy.sparse.csr_matrix). `cell_order` pandas DataFrame has columns `cell_name`, `reference` and optionally `sample`. `gene_order` pandas DataFrame has columns `gene`, `chr`, `arm`, `chr_arm`, `start`, `end`. These can be built from files with the helpers `get_cell_order()` and `get_gene_order()`.

```python
counts = adata.X
counts, cell_order = cnv.get_cell_order(cells_file, counts, adata.obs_names, column='reference', sample_col='sample')
counts, gene_order = cnv.get_gene_order(counts, adata.var_names, gtf_file, exclude_immune=True, sex_chr=False)
obj = cnv.SwiftCNV(counts, cell_order, gene_order)
```

Run the CNV estimation with advanced parameters. Gene smoothing averages the value of each gene over a window of genes: `bases_window` in MB and `genes_window` in number of genes. If both are defined the shorter window to each direction applies, to avoid smoothing over distant genes and to increase resolution if many genes are available. If both are None (default) the windows are set to 30MB and 1% of the total remaining genes (with a minimum of 51).

```python
# These are the default parameters
cnv_scores = obj.run(
    min_cells_per_gene=3,    # Filter genes expressed in less than these cells
    cutoff=0.1,              # Filter genes below this normalized expression
    bound_sd_amplifier=3.0,  # Bound values to these times the std of the matrix (in log scale)
    substract_reference_by_sample=False,  # Substract the mean of the reference by each sample
    smooth_by='arm',         # Stratify gene smoothing by 'chr' or 'arm'
    bases_window=None,       # Window in MB for smoothing
    genes_window=None,       # Window in number of genes for smoothing
    denoise=True,            # Denoise low values that are likely noise
    noise_filter=0.1,        # Threshold to consider noise (|value| < noise_filter)
    sd_amplifier=1.0,        # Threshold to consider noise (|value| < std * sd_amplifier)
    noise_logistic=True,     # Smooth denoised values with a logistic function instead of a hard filter to zero
    final_cap=1.5,           # Hard clip to [-cap, cap] to the final values
    inv_log=False,           # Apply inverse log(x + 1) to the returned matrix (centered around 1 instead of 0)
)
```

`run_from_adata` is a helper function similar to the CLI. Additional arguments are passed to the main analysis (`SwiftCNV.run`).

```python
adata = cnv.run_from_adata(
    adata,
    gtf_path='/path/to/gene_annotations.gtf.gz',
    reference_col='cell_type',
    reference_vals=['T-cell', 'Macrophages'],  
    sample_col='sample',
    read_X=True,
    exclude_immune=True,
    cutoff=0.15,
    genes_window=75,
    min_cells_per_gene=5,
)
```

If an AnnData object is provided directly as input, the function returns a new AnnData object with the cnv_scores matrix added to `adata.obsm["cnv_mat"]`. Therefore here `output_dir` is optional.


## Outputs

Output files are placed under `-o`/`--output` (or `output_dir` if defined). SwiftCNV generates 3 main files:
- `cnv_scores.npz`: compressed matrix (cells x genes) containing the CNV values
- `cell_order.tsv.gz`: file containing cell barcodes, reference status and sample_id if provided
- `gene_order.tsv.gz`: file containing gene metadata

If `-p`/`--plot` was specified:
- `cnv_scores.png`: Heatmap plot of the whole CNV scores matrix
- `cnv_scores_by_sample.pdf`: Heatmap plots of each sample if provided

if `--hmm` was specified HMM segmentation outputs will go to a `hmm/` directory:
- `cnv_states.tsv.gz`: DataFrame containing 3-states labels matrix by subcluster, cell or sample, depending on `--hmm-by`
- `cnv_states.png`: Heatmap plot with the found HMM states
- `tumor_subclusters.tsv.gz`: subcluster labels for the state HMM clustering if `--hmm-by=subcluster` (default)
