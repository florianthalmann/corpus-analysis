import numpy as np

def get_msa_entropy(msa):
    return sum([get_column_entropy(c) for c in msa])

def get_column_entropy(col):
    p = np.bincount(col)/len(col)
    p = p[np.nonzero(p)]
    return -np.sum(np.log2(p)*p)

# print(get_msa_entropy([[0,1,2,3]]))
# print(get_msa_entropy([[0,0,0,0],[0,0,0,1],[0,0,1,2]]))