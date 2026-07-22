# SwiftCNV

SwiftCNV is a fast and scalable Python implementation of the core [InferCNV](https://github.com/broadinstitute/inferCNV/wiki) algorithm to infer copy number variations (CNVs) from single-cell RNA-seq data. It provides additional features and is designed for seamless interoperability with Anndata objects and Scanpy.

## Documentation
For detailed information and example tutorials, please refer to our documentation.

## Installation

## Basic usage

SwiftCNV can be run from the command line or from a Jupyter notebook. Reference cells can be specified using a TSV file with two columns, cell_name and reference, where the reference column contains TRUE or FALSE to indicate whether each cell is used as a reference. Alternatively, reference cells can be specified by providing the column in adata.obs containing the cell type annotations (--reference-col) and which ones should be used as reference (--reference-value). The input also requires a GTF file containing gene annotation information.  This file can be easily obtained from [Gencode](https://www.gencodegenes.org/human/) database. Finally, the column identifying the samples must be specified.

### Option 1: Command line with annotation file
The following example illustrates the format of the required annotation.tsv file.

| cell_name | reference |
| --- | --- |
| AAATGCCTCACATACG | True |
| AACCATGGTTATTCTC | False |
| AACGTTGGTTTACTCT | True |
| ... | ... |

SwiftCNV can be run using the following command:

```bash
python3 cli.py \
    -i /path/to/adata.h5ad \
    -o /path/to/output \
    --reference /path/to/annotation.tsv \
    --gtf-path /path/to/gene_annotations.gtf.gz \
    --sample-col sample \
    --plot
```

### Option 2: Command line with reference cell types from adata
```bash
python3 cli.py \
    -i /path/to/adata.h5ad \
    -o /path/to/output \
    --reference-col cell_type \
    --reference-vals T-cell Macrophages \
    --gtf-path /path/to/gene_annotations.gtf.gz \
    --sample-col sample \
    --plot

```

### Option 3: Jupyter notebook

```python
import scanpy as sc
import logging
import swiftcnv as cnv
from swiftcnv.data import Qian2020_Ovarian

logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S',
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
```

```python
# Load example data
adata = Qian2020_Ovarian()

# Run SwiftCNV
adata = cnv.run_from_adata(
    adata,
    gtf_path='/path/to/gene_annotations.gtf.gz',
    reference_col='cell_type',
    reference_vals=['T-cell', 'Macrophages'],  
    sample_col='sample',
    exclude_immune=True
)

adata
```
When an AnnData object is provided, the cnv matrix is added to a new obsm layer called `cnv_mat`


## Outputs

Output files are placed under `-o`/`--output` directory or `output_dir` arg if defined. SwiftCNV generates 3 main files:

- `cnv_scores.npz`: compressed matrix (cells x genes) containing the CNV values.
- `cell_order.tsv.gz`: file containing cell barcodes, reference status and sample_id if provided.
- `gene_order.tsv.gz`: file containing gene metadata.

In addition, if `--plot` was specified:
- `cnv_scores.png`: Heatmap plot of the CNV matrix containing all the samples.
- `cnv_scores_by_sample.pdf`: Heatmap plots of each individual sample.

if `--hmm` was specified a new directory will be created called `hmm` and:
- `cnv_states.tsv.gz`: DataFrame containing 3-states labels matrix by subcluster, cell or sample, depending on `--hmm-by`
- `cnv_states.png`: Heatmap plot with the found HMM states. 
- `tumor_subclusters.tsv.gz`: subcluster labels for the state HMM clustering if `--hmm-by=subcluster` (default)
