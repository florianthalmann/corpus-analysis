from collections import defaultdict, Counter
import numpy as np
from ..features import to_multinomial
from ..alignment.smith_waterman import smith_waterman

#one layer contains repetition
def auto_labeled(hierarchy):
    hierarchy = remove_silence(hierarchy)
    if min([len(h) for h in hierarchy[0]]) == 0:#empty layers are also auto-labeled...
        return False
    h = to_int_labels(hierarchy)[1]
    merged = lambda l: l[np.hstack(([0], np.where(np.diff(l) != 0)[0]+1))]
    all_diff = lambda l: len(np.unique(l)) == len(merged(l))
    return all_diff(h[0]) != all_diff(h[1])
    #return all_diff(h[1]) or all_diff(h[0]) #top level repeats, bottom not

def remove_silence(hierarchy):
    rem = [tuple(zip(*[s for s in zip(*l) if s[1] != 'Silence']))
        for l in zip(*hierarchy)]
    rem = [r if len(r) > 0 else (np.array([]), []) for r in rem]#deal with entirely silent levels
    rem = tuple(zip(*rem))
    return (tuple([np.vstack(r) if len(r) > 0 else np.array([]) for r in rem[0]]), rem[1])

#def replace_second_silence()

#proportion of label reusage
def repetitiveness(hierarchy):
    h = to_int_labels(hierarchy)[1]
    return np.mean([(len(l)-len(np.unique(l))) / (len(l)-1) if len(l) > 1 else 1
        for l in h])

def complexity(hierarchy):
    h = to_int_labels(hierarchy)[1]
    return np.mean([len(l) for l in h])

#boolean monotonicity: all interval times of lower levels are contained in higher levels
def monotonicity(hierarchy):
    ivls = hierarchy[0]
    return all([set(np.unique(ivls[i])) >= set(np.unique(ivls[i-1]))
        for i in range(1, len(ivls))])

#mcfee/kinnard monotonicity
def label_monotonicity(hierarchy, beats):
    return pairwise_recalls(beatwise_ints(hierarchy, beats))

#monotonicity without dependencies between different same-label areas
#paper: interval monotonicity
def interval_monotonicity(hierarchy, beats):
    labels = beatwise_ints(hierarchy, beats)
    labels = np.array([relabel_adjacent(l) for l in labels])
    return pairwise_recalls(labels)

def strict_transitivity(hierarchy):
    return transitivity(hierarchy, num_identical_pairs)

def order_transitivity(hierarchy, delta=1):
    return transitivity(hierarchy, lambda c: num_similar(c, delta))

#of all parent pairs with same labels, how many child sequences are similar
def transitivity(hierarchy, sim_func):
    child_dict = to_child_dict(to_tree(hierarchy))
    num_conns = num_connections([len(cs) for cs in child_dict.values()])
    if num_conns == 0: return 1
    return sum([sim_func(c) for c in child_dict.values()]) / num_conns

#of all possible pairs of parents, how many have either different label or similar child sequences
def transitivity2(hierarchy, sim_func):
    child_dict = to_child_dict(to_tree(hierarchy))
    num_conns = num_connections([len(cs) for cs in child_dict.values()])
    if num_conns == 0: return 1
    num_sim = sum([sim_func(c) for c in child_dict.values()])
    total = num_connections([sum([len(cs) for cs in child_dict.values()])])
    return (num_sim+(total-num_conns)) / total

def to_child_dict(tree, child_dict=None):
    if not child_dict: child_dict = defaultdict(list)#default param didn't work
    for c in tree[1:]:
        if len(c) > 1:
            child_dict[c[0]].append([cc[0] for cc in c[1:]])
            to_child_dict(c, child_dict)
    return child_dict

def num_identical_pairs(list):
    list = [str(l) for l in list]
    return num_connections([c[1] for c in Counter(list).most_common()])

def num_similar(list, delta):
    num = 0
    for i in range(len(list)):
        for j in range(i+1, len(list)):
            max_len = max(len(list[i]), len(list[j]))
            #print(list[i], list[j], smith_waterman(list[i], list[j])[0])
            if len(smith_waterman(list[i], list[j])[0]) >= max_len-delta:
                num += 1
    return num

def num_connections(group_sizes):
    return sum([n*(n-1)/2 for n in group_sizes])

#of all similarly labeled pairs on each level, how many have the same parent
def pairwise_recalls(labels):
    same = [np.triu(np.equal.outer(l, l), k=1) for l in labels]
    same = [set(zip(*np.nonzero(s))) for s in same]
    return np.mean([len(same[i].intersection(same[i-1])) / len(same[i])
        for i in range(1, len(same))])

#of all possible pairs on each level, how many are either labeled differently or have the same parent
def pairwise_recalls2(labels):
    same = [np.triu(np.equal.outer(l, l), k=1) for l in labels]
    same = [set(zip(*np.nonzero(s))) for s in same]
    nc = [num_connections([len(l)]) for l in labels]
    smnc = [n-len(s) for s,n in zip(same,nc)]
    return np.mean([(len(same[i].intersection(same[i-1]))+smnc[i]) / nc[i]
        for i in range(1, len(same))])

def to_tree(hierarchy):
    #add top node
    hierarchy = list(hierarchy[0]), list(hierarchy[1])
    if len(hierarchy[0][0]) > 1:
        hierarchy[0].insert(0,
            np.array([[hierarchy[0][0][0][0], hierarchy[0][0][-1][-1]]]))
        hierarchy[1].insert(0, ['S'])
    #make monotonic so that every child node has a unique parent
    hierarchy = make_monotonic(hierarchy)
    #initialize with lowest level
    tree = [(i,l,[]) for i,l in zip(hierarchy[0][-1], hierarchy[1][-1])]
    #iteratively build higher levels
    for intervals,labels in list(reversed(list(zip(*hierarchy))))[1:]:
        for i,l in zip(intervals,labels):
            children = [n for n in tree if i[0] <= n[0][0] and n[0][1] <= i[1]]
            #didn't want to work otherwise
            tree = [n for n in tree if not (i[0] <= n[0][0] and n[0][1] <= i[1])]
            tree.append((i,l,children))
    return to_label_tree(tree[0])

#e.g. [0,[1,2],[]]
def to_label_tree(tree):
    return [tree[1]] + [to_label_tree(c) for c in tree[2]]

def make_monotonic(hierarchy):
    parent_times = ()
    output = []
    for intervals, labels in zip(*hierarchy):
        for t in parent_times:
            intervals, labels = split_level_at(t, intervals, labels)
        output.append([intervals, labels])
        parent_times = set(np.unique(intervals))
    return list(zip(*output))

def split_level_at(time, intervals, labels):
    index = next((i for i,v in enumerate(intervals) if v[0] < time < v[1]), -1)
    if index >= 0:
        s1, s2 = [intervals[index][0], time], [time, intervals[index][1]]
        intervals = np.insert(intervals, index, s1, axis=0)
        intervals[index+1] = s2
        labels = np.insert(labels, index, labels[index])
    return intervals, labels

def beatwise_ints(hierarchy, beats):
    hierarchy = to_int_labels(hierarchy)
    hierarchy = [s for s in hierarchy[0] if len(s) > 0],\
        [[int(i) for i in s] for s in hierarchy[1] if len(s) > 0]
    beat_intervals = list(zip(beats[:-1], beats[1:]))
    values = []
    for intervals, labels in zip(*hierarchy):
        values.append([])
        for b in beat_intervals:
            indices = np.where(b[0] >= intervals[:,0])[0]
            values[-1].append(labels[indices[-1]] if len(indices) > 0 else -1)
    return np.array(values)

def to_int_labels(hierarchy):
    unique = np.unique(np.concatenate(hierarchy[1]), axis=0)
    unique_index = lambda f: np.where(unique == f)[0][0]
    labels = [np.array([unique_index(f) for f in s]) for s in hierarchy[1]]
    return hierarchy[0], labels

def relabel_adjacent(labels):
    sp = np.split(labels, np.where(np.diff(labels) != 0)[0]+1)
    return np.hstack([np.repeat(i,len(s)) for i,s in enumerate(sp)])

#print(pairwise_recalls([[0,0,0,1,1,1],[2,2,3,3,4,4]]))
#print(recursive_transitivity([0,[1,[2],[3]],[1,[2]]]))#[0,[[1,[[2,[]],[3,[]]]]]]))#,(1,[4,5]),(0,[2]),(1,[4,5])]))
#print(to_child_dict([0,[1,[2],[3]],[1,[2]]]))
#print(num_similar([[0,1],[0,1,2],[0,1,2,3]], 1))
#print(strict_transitivity([0,[1,[2,[4],[5],[6]],[3]],[1,[2,[4],[5]],[3]], [2,[4]]]))
#print(order_transitivity([0,[1,[2,[4],[5],[6]],[3]],[1,[2,[4],[5]],[3]], [2,[4]]]))