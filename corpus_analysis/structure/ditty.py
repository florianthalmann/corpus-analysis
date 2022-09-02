import sys, os
sys.path.insert(1, '/Users/flo/Projects/Code/Kyoto/GraphDitty')
#import fusion muted
old_stdout = sys.stdout
sys.stdout = open(os.devnull, "w")
from SongStructure import getFusedSimilarity
sys.stdout = old_stdout

sr=22050
hop_length=512
reg_diag=1.0
reg_neighbs=0.0

def fused_matrix(filename, win_fac=-2, wins_per_block=4, K=10, niters=10):
    res = getFusedSimilarity(filename, sr, hop_length, win_fac, wins_per_block,
        K, reg_diag, reg_neighbs, niters, False, False)
    return res['Ws']['Fused'], res['times']