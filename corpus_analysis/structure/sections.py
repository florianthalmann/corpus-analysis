import numpy as np

class Section:
    def __init__(self, position, length, subdivision):
        self.p = position
        self.l = length
        self.s = subdivision
    
    def __key(self):
        return (self.p, self.l, self.s)
    
    def __repr__(self):
        return 'S('+str(self.p)+', '+str(self.l)+', '+str(self.s) + ')'
    
    def __hash__(self):
        return hash(self.__key())
    
    def __eq__(self, other):
        return self.__key() == other.__key()
    
    def end(self):
        return self.p+self.l
    
    def contains(self, other):
        return self.p <= other.p and other.end() <= self.end() and\
            other.s % self.s == 0
    
    def overlap(self, other):
        if other.s % self.s == 0 or self.s % other.s == 0:
            return (min(self.end(), other.end()) - max(self.p, other.p)) / \
                max(self.l, other.l)
        return 0
    
    def merge(self, other):
        self.p = min(self.p, other.p)
        self.l = max(self.end(), other.end())-self.p
        self.s = min(self.s, other.s)

def segments_to_sections(segments):
    return [Section(s[0][0], len(s)+s[0][1]-s[0][0], s[0][1]-s[0][0])
        for s in segments]

def remove_contained(sections):
    out = []
    for s in sorted(sections, key=lambda s: s.l, reverse=True):
        if len(out) == 0 or not any([o.contains(s) for o in out]):
            out.append(s)
        else: print([(o, s) for o in out if o.contains(s)])
    return out

def merge_overlapping(sections, min_overlap=0.9):
    out = []
    for s in sections:
        merged = False
        for i,t in enumerate(out):
            if s.overlap(t) >= min_overlap:
                t.merge(s)
                merged = True
        if not merged:
            out.append(s)
    return out
