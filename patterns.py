from heapq import merge
from functools import reduce
import numpy as np

class Pattern:
    def __init__(self, point, length, translations):
        self.p = point
        self.l = length
        self.t = translations
    
    def __repr__(self):
        return 'P('+str(self.p)+', '+str(self.l)+', '+str(self.t) + ')'
    
    def get_t_limits(self):
        return [[self.p+t, self.p+t+self.l] for t in self.t]
    
    def to_occurrences(self):
        return np.array([np.arange(l[0], l[1]) for l in self.get_t_limits()])
    
    def to_indices(self):
        return np.concatenate(self.to_occurrences())
    
    def to_boundaries(self):
        return np.unique(np.concatenate(self.get_t_limits()))
    
    def contains(self, other):
        return len(np.setdiff1d(other.to_indices(), self.to_indices())) == 0
    
    def distance(self, other):
        return min([abs((self.p+t)-(other.p+u))
            for t in self.t for u in other.t])
    
    def divide_at_relative(self, pos):
        if 0 < pos and pos < self.l:
            return [Pattern(self.p, pos, self.t),
                Pattern(self.p+pos, self.l-pos, self.t)]
        return [self]
    
    def divide_at_absolute(self, pos):
        relative_locs = np.unique([pos-l[0] for l in self.get_t_limits()
            if l[0] < pos and pos < l[1]])
        return reduce(lambda d, l: d[0].divide_at_relative(l) + d[1:],
            reversed(relative_locs), [self])

def segments_to_patterns(segments):
    return [Pattern(s[0][0], len(s), [0, s[0][1]-s[0][0]]) for s in segments]

#ignores translations beyond first two
def patterns_to_segments(patterns):
    return [np.dstack(p.to_occurrences()[:2])[0] for p in patterns]

#print(Pattern(3, 2, [0,10,18]).to_boundaries())
#print(Pattern(3, 2, [0,10,18]).to_indices())
#print(Pattern(3, 2, [0,10,18]).to_occurrences())
#print(to_segments([Pattern(3, 2, [0,10,18])]))
#print(Pattern(3, 4, [0,10,15]).contains(Pattern(3, 2, [0,10,18])))
#print(Pattern(3, 4, [0,12,15]).distance(Pattern(5, 2, [0,0,25])))
#print(Pattern(3, 4, [0,12,15]).divide_at_absolute(16))