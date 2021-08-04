import numpy as np
from ..util import flatten, indices_of_subarray
from .similarity import isect_similarity, sw_similarity
from .hierarchies import to_hierarchy

#returns all recursive parts of the given sequence
def parts(seq, sec, unique=True):
    p = [sec[s] for s in seq if s in sec]
    if len(p) == 0: return np.array([], dtype=int)
    p = np.concatenate(p)
    pp = parts(p, sec, unique)
    c = np.concatenate((p, pp))
    return np.unique(c) if unique else c

#split up at recurring sections and remove duplicates
def split_at_recurring_secs(sequences, sections):
    flat = np.hstack(sequences)
    rec, n = np.unique(flat, return_counts=True)
    #print(rec, n)
    #indices of locations of recurring sections
    locs = [np.nonzero(np.isin(s, rec[n > 1]))[0] for s in sequences]
    #print(locs)
    #split before and after recurring secs
    subs = flatten([np.split(sequences[i], np.unique(np.hstack((l, l+1))))
        for i,l in enumerate(locs)], 1)
    subs = [s for s in subs if len(s) > 0]
    #print(subs)
    #remove duplicates of recurring sections
    u, keep = np.unique([tuple(s) for s in subs], return_index=True)
    #print(u, keep)
    return [subs[i] for i in np.sort(keep)]

def unpack_sections(sequences, sections):
    return [flatten([list(sections[e]) if e in sections else e for e in s])
        for s in sequences]

def flat_seq(sequence, sections):
    return np.array(flatten(to_hierarchy(sequence, sections)))

#true if a2 is a subarray of a1
def subarray(a1, a2):
    if len(a1) >= len(a2): return len(indices_of_subarray(a1, a2)) > 0

def contained(s1, s2):
    return ((subarray(s1[0], s2[0]) and subarray(s1[1], s2[1]))
        or (subarray(s1[0], s2[1]) and subarray(s1[1], s2[0])))

def find_dependent(similars, sections):
    flats = [(flat_seq(s[1][0], sections), flat_seq(s[1][1], sections)) for s in similars]
    return [(i,j) for i in range(len(similars)) for j in range(len(similars))
        if contained(flats[i], flats[j])]

def rec_find_similars(sequences, sections, min, flat=False):
    split = split_at_recurring_secs(sequences, sections)
    #print(split)
    splitp = [np.sort(parts(s, sections, False)) for s in split]
    #print(splitp)
    if flat:
        flats = [flat_seq(s, sections) for s in sequences]
        sims = [(sw_similarity(flats[i], flats[j]), (i,j))
            for i in range(len(flats)) for j in range(len(flats))[i+1:]]
    else:
        sims = [(isect_similarity(splitp[i], splitp[j]), (i,j))
            for i in range(len(splitp)) for j in range(len(splitp))[i+1:]]
    # best = sorted(sims, key=lambda s: s[0], reverse=True)[0]
    # print(best)
    best = [s for s in sims if s[0] >= min]
    #filter out direct and indirect containments
    best = [(p,(i,j)) for (p,(i,j)) in best if len(np.intersect1d(split[i], split[j])) == 0]
    best = [(p,(i,j)) for (p,(i,j)) in best
        if (len(split[i]) > 1 or len(np.intersect1d(split[i], splitp[j])) == 0)
        and (len(split[j]) > 1 or len(np.intersect1d(split[j], splitp[i])) == 0)]
    # best = [(p,(i,j)) for (p,(i,j)) in best
    #     if len(np.intersect1d(split[i], splitp[j])) == 0
    #     and len(np.intersect1d(split[j], splitp[i])) == 0]
    print(best)
    best = [(p, (split[i], split[j])) for (p,(i,j)) in best]
    
    #filter out indirect containments
    #best = [s for s in sims if ]
    #plot_matrix(sims)
    #unpack sections not occurring at lower levels
    to_unpack = [s for s in np.hstack(split) if s not in np.hstack(splitp)]
    #print(to_unpack)
    to_unpack = {k:v for k,v in sections.items() if k in to_unpack}
    #print(list(to_unpack.keys()))
    #recur until all unpacked
    if len(to_unpack) > 0:
        unpacked = unpack_sections(split, to_unpack)
        subsims = rec_find_similars(unpacked, sections, min)
        return best+subsims
    return best

def find_similars(sequences, sections, min=0.5):
    sims = rec_find_similars(sequences, sections, min)
    #sort and remove duplicates
    sims = [(p,(s,t)) if len(s) < len(t) else (p,(t,s)) for (p,(s,t)) in sims]
    print(sims)
    u,id = np.unique([str((p,(tuple(s),tuple(t)))) for (p,(s,t)) in sims], return_index=True)
    sims = [sims[i] for i in np.sort(id)]
    print(sims)
    dep = find_dependent(sims, sections)
    print(dep)
