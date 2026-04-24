import numpy as np

SHAPE_MAX_RENDER = 200

def is_symmetric(A, tol=1e-8):
    return np.allclose(A, A.T, atol=tol)
