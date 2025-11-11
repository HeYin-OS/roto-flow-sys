import numpy as np
from scipy import linalg
A = np.random.randn(4,4)
w, v = linalg.eig(A)
print("Eigenvalues:", w)