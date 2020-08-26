from heapq import merge
import numpy as np

class Pattern:
    def __init__(self, point, length, translations):
        self.p = point
        self.l = length
        self.t = translations
    
    def to_indices(self):
        return np.concatenate([np.arange(self.p+t, self.p+t+self.l)
            for t in [0]+self.t])
    
    def contains(self, other):
        return len(np.setdiff1d(other.to_indices(), self.to_indices())) == 0
    
    def distance(self, other):
        return min([abs((self.p+t)-(other.p+u))
            for t in [0]+self.t for u in [0]+other.t])

def to_patterns(segments):
    return [Pattern(s[0][0], len(s), s[0][1]) for s in segments]

def to_segments(patterns):
    return []

#print(Pattern(3, 4, [10,15]).contains(Pattern(3, 2, [10,18])))
#print(Pattern(3, 4, [12,15]).distance(Pattern(5, 2, [20,25])))