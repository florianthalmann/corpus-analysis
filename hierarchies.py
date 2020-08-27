import math
from functools import reduce
from patterns import Pattern, segments_to_patterns, patterns_to_segments

# returns the min dist between p and any parents of p in patterns
def min_dist_from_parents(p, patterns):
    parents = [q for q in patterns if q.contains(p)]
    if len(parents) > 0:
        return min([p.distance(q) for q in parents])
    return math.inf

# filter and sort list of patterns based on given params
def filter_and_sort_patterns(patterns, min_len, min_dist, parents):
    #filter patterns that are too short or too close to parents
    min_dists = [min_dist_from_parents(p, parents) for p in patterns]
    filtered = [p for i,p in enumerate(patterns)
        if p.l >= min_len and min_dists[i] >= min_dist]
    #sort by position and smallest vector
    secondary = sorted(filtered, key=lambda p: (p.p, min(p.t)))
    #reverse sort by min(dist from parents, length)
    return sorted(secondary, key=
        lambda p: min(min_dists[patterns.index(p)], p.l), reverse=True)

# removes any pattern overlaps, starting with longest pattern,
# adjusting shorter ones to fit within limits
def remove_overlaps(patterns, min_len, min_dist):
    result = []
    patterns = filter_and_sort_patterns(patterns, min_len, min_dist, result)
    print(patterns)
    while len(patterns) > 0:
        next = patterns.pop(0)
        result.append(next)
        new_boundaries = next.to_boundaries()
        for b in new_boundaries:
            patterns = [q for p in patterns for q in p.divide_at_absolute(b)]
        patterns = filter_and_sort_patterns(patterns, min_len, min_dist, result)
    return result

def add_transitivity(patterns):
    
    return 

def make_hierarchical(segments, min_len, min_dist):
    patterns = segments_to_patterns(segments)
    patterns = remove_overlaps(patterns, min_len, min_dist)
    return patterns_to_segments(patterns)

print(remove_overlaps([Pattern(3, 4, [10,15]), Pattern(2, 5, [10,18])], 2, 2))

