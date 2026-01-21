import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm