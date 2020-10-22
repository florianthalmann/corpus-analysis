import math
from collections import OrderedDict
from heapq import merge
from functools import reduce
import numpy as np
import sortednp as snp

class Pattern:
    def __init__(self, point, length, translations):
        self.p = point
        self.l = length
        self.t = translations
    
    def __key(self):
        return (self.p, self.l, tuple(self.t))
    
    def __repr__(self):
        return 'P('+str(self.p)+', '+str(self.l)+', '+str(self.t) + ')'
    
    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()
    
    #new_t need to be absolute positions
    def add_new_translations(self, new_t, non_overlapping=False):
        absolute = np.unique(np.sort([self.p+t for t in self.t] + new_t))
        #if len(new_t) > 0: print(self, new_t, absolute, [a-absolute[0] for a in absolute])
        if non_overlapping:
            absolute = [a for i,a in enumerate(absolute)
                if i == 0 or a-absolute[i-1] >= self.l]
        self.p = absolute[0] #move ref point p
        self.t = [a-self.p for a in absolute] #update relative translations
    
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
    
    def overlaps(self, other):
        #print(other.to_indices(), self.to_indices())
        #print(snp.intersect(other.to_indices(), self.to_indices()))
        return len(snp.intersect(other.to_indices(), self.to_indices())) > 0
    
    def distance(self, other):
        return min([abs(t-u) for t in self.t for u in other.t])
    
    #removes all occurrences that are closer than min_dist from any of the given
    #reference patterns' occurrences. returns the remaining minimum distance from
    #any reference
    def remove_close_occs(self, ref_segs, min_dist):
        segs = self.to_segments()
        to_be_removed = []
        min_dist_from_refs = math.inf
        for i,t in enumerate(segs):
            #NOT CALC IF T ALREADY TO BE REMOVED!!!
            dists = [segment_dist(t,u) for u in ref_segs]
            #BREAK DIST CALC WHEN NONZERO < MIN_DIST
            nonzero = [d for d in dists if d > 0]
            if len(nonzero) > 0:
                min_nonzero = min(nonzero)
                if min_nonzero < min_dist:
                    to_be_removed.append(seg_index_to_t_index(i, len(self.t)))
                else:
                    min_dist_from_refs = min(min_dist_from_refs, min_nonzero)
        self.t = [t for i,t in enumerate(self.t) if i not in to_be_removed]
        return min_dist_from_refs
    
    def first_occ_overlaps(self, other):
        return len(snp.kway_intersect(np.arange(self.p, self.p+self.l),
            np.arange(other.p, other.p+other.l))) > 0
    
    #returns relative positions of occurrences of other in self where at least
    #the given proportion of other is in self
    def internal_positions(self, other, proportion=1):
        return np.sort(np.unique([o[0]-s[0]
            for s in self.to_occurrences() for o in other.to_occurrences()
            if len(snp.intersect(s, o)) >= len(o)*proportion]))
    
    #returns triples (p,s,l) for each position p at which a segment of other
    #starting at s with length l occurs in an occurrence of self
    def partial_appearances(self, other):
        appearances = []
        for s in self.to_occurrences():
            for o in other.to_occurrences():
                i = snp.intersect(s, o)
                if len(i) > 0:
                    if i[0] == o[0]: #beginning contained
                        appearances.append((o[0]-s[0], 0, len(i)))
                    else: #end contained
                        appearances.append((0, other.l-len(i), len(i)))
        return list(OrderedDict.fromkeys(appearances)) #unique
    
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
    
    def to_segments(self):
        occs = self.to_occurrences()
        return [np.dstack([o1,o2])[0] for i,o1 in enumerate(occs)
            for o2 in occs[i+1:]]

def segments_to_patterns(segments):
    return [Pattern(s[0][0], len(s), [0, s[0][1]-s[0][0]]) for s in segments]

#ignores translations beyond first two
def patterns_to_segments(patterns):
    return [s for p in patterns for s in p.to_segments()]

#returns the distance between s1 and s2 if they overlap, else inf
def segment_dist(s1, s2):
    #s1, s2 = sorted([s1, s2], key=lambda s: s[0][0])
    if s2[0][0] < s1[0][0]:
        s1, s2 = s2, s1
    if s2[0][0] < s1[-1][0]:
        return abs((s1[0][1]-s1[0][0])-(s2[0][1]-s2[0][0]))
    else: return math.inf

def seg_index_to_t_index(i, num_t):
    return [j for i in range(num_t) for j in range(i+1,num_t)][i]

#print(Pattern(3, 2, [0,10,18]).to_boundaries())
#print(Pattern(3, 2, [0,10,18]).to_indices())
#print(Pattern(3, 2, [0,10,18]).to_occurrences())
#print(to_segments([Pattern(3, 2, [0,10,18])]))
#print(Pattern(3, 4, [0,10,15]).contains(Pattern(3, 2, [0,10,18])))
#print(Pattern(3, 4, [0,12,15]).distance(Pattern(5, 2, [0,0,25])))
#print(Pattern(3, 4, [0,12,15]).divide_at_absolute(16))
#print(Pattern(3, 4, [0,12,15]).internal_positions(Pattern(5, 2, [0,0,25])))