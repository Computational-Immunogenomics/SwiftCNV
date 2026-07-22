import scanpy as sc
from anndata import AnnData
from importlib.resources import files, as_file


def Qian2020_Ovarian() -> AnnData:
	'''
	Derived from :cite:`https://www.weizmann.ac.il/sites/3CA/ovarian`.
	'''
	resource = files('swiftcnv.data').joinpath('datasets', 'Qian2020_Ovarian.h5ad')
	with as_file(resource) as filepath:
		return sc.read_h5ad(filepath)
