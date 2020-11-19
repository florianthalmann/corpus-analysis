import sys, json
import numpy as np
from collections import deque, defaultdict
from functools import reduce

#rewrite using numpy: https://core.ac.uk/download/pdf/288487378.pdf

def merge_sorted_arrays(a1, a2):
    result = deque()
    a1 = deque(a1)
    a2 = deque(a2)
    while a1 and a2:
        if tuple(a1[0][0]) < tuple(a2[0][0]):
            result.append(a1.popleft())
        #not sure if this is necessary
        elif tuple(a1[0][0]) == tuple(a2[0][0]) and tuple(a1[0][1]) < tuple(a2[0][1]):
            result.append(a1.popleft())
        else:
            result.append(a2.popleft())
    result += a1
    result += a2
    return np.array(result)

def intersect_sorted_arrays(a1, a2):
    result = deque()
    a1 = deque(a1)
    a2 = deque(a2)
    while a1 and a2:
        if tuple(a1[0]) == tuple(a2[0]):
            result.append(a1.popleft())
            a2.popleft()
        elif tuple(a1[0]) < tuple(a2[0]):
            a1.popleft()
        else:
            a2.popleft()
    return np.array(result)

def group_by_first(input):
    result = defaultdict(list)
    for k, v in input:
        result[str(k)].append(v)
    return result

def json_pattern(points, vecs, occs):
    return {
        'points': [p.tolist() for p in points],
        'vectors': [v.tolist() for v in vecs],
        'occurrences': np.array(occs).tolist()#[o.tolist() for o in occs],
    }

def to_json(points, pats, vecs, occs):
    patterns = [json_pattern(p, vecs[i], occs[i]) for i, p in enumerate(pats)]
    return {'points': points.tolist(), 'patterns': patterns}

def siatec(points, savepath):
    print 'table'
    points = np.unique(points, axis=0)
    vector_table = [[(q - p, p) for q in points] for p in points]
    half_table = [r[i+1:] for i, r in enumerate(vector_table) if i < len(r)-1]
    print 'merge'
    table_list = reduce(merge_sorted_arrays, half_table)
    print 'group'
    patterns = group_by_first(table_list).values()
    pdict = {str(p): i for i, p in enumerate(points)}
    simple_table = [[r[0] for r in c] for c in vector_table]
    tsls = [[simple_table[pdict[str(o)]] for o in p] for p in patterns]
    print 'intersect'
    vectors = [reduce(intersect_sorted_arrays, ts) for ts in tsls]
    print 'json'
    occurrences = [[p + v for p in patterns[i]] for i, v in enumerate(vectors)]
    result = to_json(points, patterns, vectors, occurrences)
    print 'save'
    with open(savepath, 'w') as outfile: json.dump(result, outfile)

siatec(np.array(json.loads(sys.argv[1])), sys.argv[2])