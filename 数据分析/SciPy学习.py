import numpy as np
import scipy
from scipy import sparse

"""
    SciPy是Python 中用于科学计算的函数集合。它具有线性代数高级程序、数学函数优化、
    信号处理、特殊数学函数和统计分布等多项功能。
"""

eye = np.eye(4)
print("Numpy array: \n{}".format(eye))
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))