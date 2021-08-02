import numpy as np
from similarity import isect_similarity
from util import flatten

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

def find_similars(sequences, sections, min=0.5):
    split = split_at_recurring_secs(sequences, sections)
    #print(split)
    splitp = [np.sort(parts(s, sections, False)) for s in split]
    #print(splitp)
    sims = [(isect_similarity(splitp[i], splitp[j]), (split[i],split[j]))
        for i in range(len(splitp)) for j in range(len(splitp))[i+1:]]
    best = sorted(sims, key=lambda s: s[0], reverse=True)[0]
    print(best)
    best = [s for s in sims if s[0] >= min]
    #plot_matrix(sims)
    #unpack sections not occurring at lower levels
    to_unpack = [s for s in np.hstack(split) if s not in np.hstack(splitp)]
    #print(to_unpack)
    to_unpack = {k:v for k,v in sections.items() if k in to_unpack}
    #print(list(to_unpack.keys()))
    if len(to_unpack) > 0:
        unpacked = unpack_sections(split, to_unpack)
        subsims = find_similar(unpacked, sections)
        return best+subsims
    return best
    #recur until all unpacked