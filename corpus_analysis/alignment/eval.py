from ..stats.util import entropy

def get_msa_entropy(msa):
    return sum([entropy(c) for c in msa])

# print(get_msa_entropy([[0,1,2,3]]))
# print(get_msa_entropy([[0,0,0,0],[0,0,0,1],[0,0,1,2]]))