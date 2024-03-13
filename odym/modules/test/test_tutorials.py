import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor
import os
import glob
    

@pytest.mark.parametrize("notebook", [x for x in glob.glob("docs/*.ipynb")] ) #
def test_notebook_exec(notebook):
    # copied from
    # https://stackoverflow.com/questions/70671733/testing-a-jupyter-notebook
    with open(os.path.join(os.getcwd(), notebook)) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        try:
            assert ep.preprocess(nb) is not None, f"Got empty notebook for {notebook}"
        except Exception:
            assert False, f"Failed executing {notebook}"