import numpy as np
from ..util import flatten, indices_of_subarray, plot_sequences
from .similarity import isect_similarity, sw_similarity, multi_jaccard
from .hierarchies import to_hierarchy

#returns all recursive parts of the given sequence
def parts(seq, sec, unique):
    p = [sec[s] for s in seq if s in sec]
    if len(p) == 0: return np.array([], dtype=int)
    p = np.concatenate(p)
    pp = parts(p, sec, unique)
    c = np.concatenate((p, pp))
    return np.unique(c) if unique else c

def all_parts(seq, sec, unique=True):
    return np.unique(np.concatenate([seq, parts(seq, sec, unique)]))

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
    s11in21, s11in22 = subarray(s1[0], s2[0]), subarray(s1[0], s2[1])
    s12in21, s12in22 = subarray(s1[1], s2[0]), subarray(s1[1], s2[1])
    return ((s11in21 and s12in22 and (not s11in22 or not s12in21))
        or (s11in22 and s12in21 and (not s11in21 or not s12in22)))

def find_dependent(similars, sections):
    flats = [(flat_seq(s[1][0], sections), flat_seq(s[1][1], sections)) for s in similars]
    plot_sequences([np.hstack([f[0], np.repeat(-1, 5), f[1]]) for f in flats], 'sims.png')
    print(flats)
    #print([(np.sort(all_parts(s[1][0], sections, False)), np.sort(all_parts(s[1][1], sections, False))) for s in similars])
    return [(i,j) for i in range(len(similars)) for j in range(len(similars))
        if contained(flats[i], flats[j])]

def rec_find_similars(sequences, sections, min, flat=True):
    seqparts = [np.sort(parts(s, sections, False)) for s in sequences]
    #print(seqparts)
    if flat:
        flats = [flat_seq(s, sections) for s in sequences]
        #print(flats)
        sims = [(sw_similarity(flats[i], flats[j]), (i,j))
            for i in range(len(flats)) for j in range(len(flats))[i+1:]]
        #print('YOOO', sims)
    else:
        sims = [(isect_similarity(seqparts[i], seqparts[j]), (i,j))
            for i in range(len(seqparts)) for j in range(len(seqparts))[i+1:]]
    # best = sorted(sims, key=lambda s: s[0], reverse=True)[0]
    # print(best)
    best = [s for s in sims if s[0] >= min]
    #filter out direct and indirect containments
    # best = [(p,(i,j)) for (p,(i,j)) in best if len(np.intersect1d(sequences[i], sequences[j])) == 0]
    best = [(p,(i,j)) for (p,(i,j)) in best
        if (len(sequences[i]) > 1 or len(np.intersect1d(sequences[i], seqparts[j])) == 0)
        and (len(sequences[j]) > 1 or len(np.intersect1d(sequences[j], seqparts[i])) == 0)]
    # best = [(p,(i,j)) for (p,(i,j)) in best
    #     if len(np.intersect1d(sequences[i], seqparts[j])) == 0
    #     and len(np.intersect1d(sequences[j], seqparts[i])) == 0]
    #print(best)
    best = [(p, (sequences[i], sequences[j])) for (p,(i,j)) in best]
    
    #filter out indirect containments
    #best = [s for s in sims if ]
    #plot_matrix(sims)
    #unpack sections not occurring at lower levels
    to_unpack = [s for s in np.hstack(sequences) if s not in np.hstack(seqparts)]
    #print(to_unpack)
    to_unpack = {k:v for k,v in sections.items() if k in to_unpack}
    #recur until all unpacked
    if len(to_unpack) > 0:
        unpacked = unpack_sections(sequences, to_unpack)
        split = split_at_recurring_secs(unpacked, sections)
        #print(split)
        subsims = rec_find_similars(split, sections, min)
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

