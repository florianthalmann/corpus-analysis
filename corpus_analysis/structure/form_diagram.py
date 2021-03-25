from collections import defaultdict, Counter
import numpy as np
from graph_tool import topology, GraphView
from ..util import plot_matrix, plot_graph, flatten
from .graphs import graph_from_matrix, prune_isolated_vertices

def sequence_probability(sequences, form_diagram, seq_to_v, probs):
    probabilities = np.empty(len(sequences))
    for i,s in enumerate(sequences):
        vs = seq_to_v[s]
        edges = [form_diagram.edge(vs[i], vs[i+1]) for i in range(len(vs)-1)]
        ps = np.array([probs.a[form_diagram.edge_index[e]] for e in edges])
        probabilities[i] = np.prod(ps)
    print((probabilities/np.max(probabilities)*1000).round())

def essence(sequences, form_diagram, v_to_seq, probs):
    print(sequences[:4])
    probs.a = -np.log(probs.a)
    size = form_diagram.num_vertices()
    start = form_diagram.vertex(size-2)
    end = form_diagram.vertex(size-1)
    vs, es = topology.shortest_path(form_diagram, start, end, probs)
    vs = np.array([int(v) for v in vs])[1:-1]
    print(v_to_seq[vs])

def form_diagram(sequences):
    sequences = [s[s >= 0] for s in sequences] #ignore gaps (-1)
    #calculate transition matrix for all section types
    size = np.max(np.concatenate(sequences))+1
    freqs = np.bincount(np.concatenate(sequences))
    ids = np.arange(size)
    transitions = np.zeros((size+2, size+2))
    for s in sequences:
        transitions[-2][s[0]] += 1 #start
        for i in range(len(s)-1):
            transitions[s[i]][s[i+1]] += 1
        transitions[s[-1]][-1] += 1 #end
    #remove isolated vertices
    isolated = np.logical_and(np.all(np.equal(transitions, 0), axis=1),
        np.all(np.equal(transitions.T, 0), axis=1))
    transitions = transitions[~isolated].T[~isolated].T
    freqs = freqs[~isolated[:-2]]
    ids = ids[~isolated[:-2]]
    #normalize transition probabilities for each section
    sums = np.sum(transitions, axis=1)
    sums[sums == 0] = 1
    sums = np.repeat(sums, transitions.shape[1]).reshape(transitions.shape)
    transitions /= sums
    plot_matrix(transitions, 'form1.png')
    #create graph
    graph, probs = graph_from_matrix(transitions, directed=True)
    # vsize = graph.new_vp("int")
    # vsize.a = freqs
    # graph = GraphView(graph, vsize.a > 3)
    edge_width = probs.copy()
    edge_width.a *= 10
    plot_graph(graph, 'form2.png', edge_width)
    
    #simplify_graph(graph)
    seq_to_v = np.repeat(-1, size)
    seq_to_v[ids] = np.arange(len(ids))
    sequence_probability(sequences, graph, seq_to_v, probs)
    essence(sequences, graph, ids, probs)

def collapse_similar(sequences):
    return

def simplify_graph(graph):
    #gather alternative paths
    forks = [v for v in graph.vertices() if len(list(v.out_edges())) > 1]
    joins = [v for v in graph.vertices() if len(list(v.in_edges())) > 1]
    print([int(f) for f in forks])
    print([int(j) for j in joins])#, len(joins))
    #print(get_alt_paths(forks[14], graph))
    paths = flatten([get_simple_alternative_paths(f, graph) for f in forks], 1)
    #paths = [p for p in paths if len(p) > 0]
    print(paths)
    print(graph)
    to_remove = set()
    for ps in paths:
        if any([len(p) == 2 for p in ps]):#remove detour
            print(ps)
            detour = next(p for p in ps if len(p) > 2)
            to_remove.update(detour[1:-1])
    for v in sorted(to_remove, reverse=True):
        graph.remove_vertex(v)
    print(graph)
    plot_graph(prune_isolated_vertices(graph), 'form3.png', weights)
    #print([len(list(p)) for p in paths])
    #print(len(paths))
    # for p in paths[0]:
    #     print(p)
    # print(paths[0][0])
    # print([p for p in paths[0]])

#returns paths between fork and the earliest vertex at which some of the forked paths join
def get_simple_alternative_paths(fork, graph):
    neighbors = [n for n in fork.out_neighbors()]
    paths = [find_paths_until_join(n1, n2, graph)
        for i,n1 in enumerate(neighbors[:-1]) for n2 in neighbors[i+1:]]
    #return [topology.all_paths(graph, fork, graph.vertex(j)) for j in joins if j != None]
    return [[[int(fork)]+p for p in ps] for ps in paths if ps != None]
    
    # current = [[v] for v in fork.out_neighbors()]
    # time = 1
    # times = defaultdict(list)
    # prev_times = len(times)
    # [[times[v].append(time) for v in c] for c in current]
    # reached = [set([int(v) for v in c]) for c in current]
    # while no_pairs_intersect(reached) and len(times) > prev_times:
    #     current = [flatten([list(v.out_neighbors()) for v in c]) for c in current]
    #     prev_times = len(times)
    #     time += 1
    #     [[times[v].append(time) for v in c] for c in current]
    #     [r.update([int(v) for v in current[i]]) for i,r in enumerate(reached)]
    #     print(reached)
    # counts = Counter([v for r in reached for v in r])
    # joins = [graph.vertex(v) for v,i in counts.items() if i > 1]
    # ratings = [sum(times[j]) for j in joins]
    # earliest_join = joins[np.argmin(ratings)]
    # print(int(fork), [int(j) for j in joins])
    # print(ratings)
    # print(whatever)
    # return topology.all_paths(graph, fork, earliest_join)

def find_paths_until_join(v1, v2, graph):
    paths = [[int(v1)], [int(v2)]]
    prev_len = 0
    while len(np.unique(flatten(paths))) > prev_len:
        prev_len = len(np.unique(flatten(paths)))
        neighbors = [list(graph.vertex(p[-1]).out_neighbors()) for p in paths]
        [paths[i].append(int(n[0])) for i,n in enumerate(neighbors) if len(n) == 1]
        join = set.intersection(*[set(p) for p in paths])
        #print(paths, join)
        if len(join) > 0:
            return [p[:p.index(list(join)[0])+1] for p in paths]
