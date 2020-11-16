from itertools import product, chain, groupby
from functools import reduce
from collections import Counter, OrderedDict
import numpy as np
import sortednp as snp
from graph_tool.all import Graph, graph_draw, GraphView, edge_endpoint_property,\
    remove_parallel_edges
from graph_tool.topology import label_components
from graph_tool.spectral import adjacency
from graph_tool.inference.blockmodel import BlockState
from graph_tool.topology import transitive_closure, all_paths, max_cliques
from graph_tool.util import find_edge
from ..util import group_adjacent, plot, plot_matrix

def adjacency_matrix(graph, weight=None):
    return adjacency(graph, weight).toarray().T

def prune_isolated_vertices(g):
    return GraphView(g, vfilt=g.get_total_degrees(g.get_vertices()) > 0)

def graph_from_matrix(matrix):
    g = Graph(directed=False)
    g.add_vertex(len(matrix))
    g.add_edge_list(list(zip(*np.nonzero(matrix))))
    #graph_draw(g, output_size=(1000, 1000), output="results/structure.pdf")
    return g

def alignment_graph(lengths=[], pairings=[], alignments=[]):
    #print('making graph')
    g = Graph(directed=False)
    seq_index = g.new_vertex_property("int")
    time = g.new_vertex_property("int")
    #add vertices
    g.add_vertex(sum(lengths))
    seq_index.a = np.concatenate([np.repeat(i,l) for i,l in enumerate(lengths)])
    time.a = np.concatenate([np.arange(l) for l in lengths])
    #add edges (alignments)
    alm_index = g.new_edge_property("int")
    seg_index = g.new_edge_property("int")
    for i,a in enumerate(alignments):
        if len(a) > 0:
            j, k = pairings[i]
            pairs = np.concatenate(a, axis=0)
            indicesJ = (np.arange(lengths[j]) + sum(lengths[:j]))[pairs.T[0]]
            indicesK = (np.arange(lengths[k]) + sum(lengths[:k]))[pairs.T[1]]
            seg_indices = np.concatenate([np.repeat(i, len(a))
                for i,a in enumerate(a)])
            g.add_edge_list(np.vstack([indicesJ, indicesK,
                np.repeat(i, len(pairs)), seg_indices]).T,
                eprops=[alm_index, seg_index])
    #g.add_edge_list([(b, a) for (a, b) in g.edges()])
    #print('created alignment graph', g)
    #g = prune_isolated_vertices(g)
    #print('pruned alignment graph', g)
    #g = transitive_closure(g)
    #graph_draw(g, output_size=(1000, 1000), output="results/casey_jones_bars.pdf")
    return g, seq_index, time, alm_index, seg_index

#combine a sequence of potential segment combinations into a consensus
#(individual seg_combos have to be sorted)
#can easily be made into beam search to speed up
def integrate_segment_combos(seg_combos, incomp_segs):
    seg_combos = [s for s in seg_combos if len(s) > 0]
    sets = []
    ratings = []
    subset_buffer = {}
    positions = {}
    for k,sc in enumerate(seg_combos):
        if len(sets) == 0:
            sets = [c for c in sc]
            ratings = np.array([len(c) for c in sc])
            for i,c in enumerate(sc):
                key = str(list(c))
                positions[key] = i
                subset_buffer[key] = np.array([], dtype=int)
        else:
            for c in sc:
                key = str(list(c))
                lenc = len(c)
                exists = key in positions
                #keep a buffer of all the locations where c a subset
                subset_locs = subset_buffer[key] if exists \
                    else np.array([], dtype=int)
                #calculate intersections where c not a known subset
                isects = [c if len(subset_locs) > i and subset_locs[i] > 0
                    else snp.intersect(c, s) for i,s in enumerate(sets)]
                ilengths = np.array([len(i) for i in isects])
                #update locations where c subset of existing sets
                subset_locs = np.concatenate((subset_locs,
                    ilengths[len(subset_locs):] == lenc))
                subset_buffer[key] = subset_locs
                #update ratings where c subset
                np.add.at(ratings, np.nonzero(subset_locs), lenc)
                #add set if not added before and if not a subset of another
                if not exists and not np.any(subset_locs):
                    #get max rating at locations where c superset
                    setlengths = np.array([len(s) for s in sets])
                    superset_locs = np.nonzero(ilengths == setlengths)[0]
                    max_rating = lenc
                    if len(superset_locs) > 0:
                        max_rating += np.max(ratings[superset_locs])
                    #append c to sets with appropriate rating
                    sets.append(c)
                    subset_buffer[key] = np.array([], dtype=int)
                    ratings = np.append(ratings, [max_rating])
                    #append union if possible
                    for k,i in enumerate(isects):
                        #neither is a subset of the other
                        if len(i) < len(c) and len(i) < len(sets[k]):
                            union = np.unique(np.concatenate([c, sets[k]]))
                            ukey = str(list(union))
                            #union not added yet and no incompatible segments
                            if not ukey in positions and\
                                    not any(len(snp.intersect(j, union)) > 1 for j in incomp_segs):
                                positions[ukey] = len(sets)
                                sets.append(union)
                                subset_buffer[ukey] = np.array([], dtype=int)
                                #rating of union = sum - rating of intersection
                                rating = max_rating+ratings[k]
                                ikey = str(list(i))
                                if ikey in positions:
                                    rating -= ratings[positions[ikey]]
                                ratings = np.append(ratings, [rating])
        #print(k, len(sc), len(sets), len(ratings), max(ratings) if len(ratings) > 0 else 0, sum(ratings) if len(ratings) > 0 else 0)
    #[print(ratings[i], s) for i, s in enumerate(sets)]
    return sets, ratings

def get_incompatible_segs(g, seg_index, out_edges):
    incomp = []
    for v in g.get_vertices():
        for vs in group_adjacent(sorted(g.get_out_neighbors(v))):
            edges = out_edges[v][np.where(np.isin(out_edges[v][:,1], vs))][:,2]
            incomp.append(list(np.unique(seg_index.a[edges])))
    #keep only unique supersets
    incomp = [k for k,_ in groupby(sorted(incomp))]
    incomp = list(filter(lambda f: not any(set(f) < set(g) for g in incomp), incomp))
    return [np.array(k) for k in incomp if len(k) > 1]

def get_edge_combos(g):
    out_edges = [g.get_out_edges(v, [g.edge_index]) for v in g.get_vertices()]
    edge_combos = []
    #for each vertex find best combinations of neighbors 
    for v in g.get_vertices():#[49:50]:
        n = sorted(g.get_out_neighbors(v))
        #split into connected segments
        vertex_combos = list(product(*group_adjacent(sorted(n))))
        edge_combos.append([])
        for c in vertex_combos:
            #collect internal edges of subgraph
            vertices = [v]+list(c)
            edges = np.concatenate([out_edges[v] for v in vertices])
            shared = edges[np.where(np.isin(edges[:,1], vertices))]
            edge_combos[-1].append(sorted(shared[:,2]))
            # filt = g.new_vertex_property("bool")
            # filt.a[[v]+list(c)] = True
            # gg = GraphView(g, vfilt=filt)
            # #check if elements of n in same clique
            # #cliques = list(max_cliques(gg))
            # #samecli = any(len(np.intersect1d(n, c)) == len(n) for c in cliques)
            # #or more simply, rate by how many edges there are
            # edge_combos[-1].append([e[2] for e in gg.get_edges([g.edge_index])])
            
            #if samecli:
            #    votes.append(np.array(n) - v)
            #    reduced.add_edge_list([(a,b) for i,a in enumerate(n) for b in n[i+1:]])
            #graph_draw(gg, vertex_text=g.vertex_index, output_size=(1000, 1000),
            #    output="results/clean_upp"+str(i)+".pdf")
    return edge_combos

def remove_subarrays(arrays, proper=True):
    arrays = sorted(arrays, key=lambda a: len(a))
    to_remove = []
    for i,a in enumerate(arrays):
        #bs = [b for b in arrays[i+1:] if len(b) < len(a)] if proper else arrays[i+1:]
        for b in arrays[i+1:]:
            if len(snp.intersect(a, b)) == len(a):
                to_remove.append(i)
    return [a for i,a in enumerate(arrays) if i not in to_remove]

def is_subarray(array, arrays):
    for a in arrays:
        if len(snp.intersect(array, a)) == len(a):
            return True
    return False

def sorted_unique(array):
    return snp.merge(array, np.array([]), duplicates=snp.DROP)

#no incompatible nodes
def validate_quasi_clique(quasi_clique, validated, incomp):
    if quasi_clique not in validated:
        incomps = np.array([incomp[u] for u in quasi_clique])
        validated[quasi_clique] = len(sorted_unique(incomps)) == len(incomps)

#TODO change to breadth first and see... probably faster!
def quasi_clique_expansions(quasi_cliques, cliques, incomp):
    max_expansions = set()
    temp_expansions = set([tuple(q) for q in quasi_cliques])
    expanded = set()
    valid = dict()
    while len(temp_expansions) > 0:
        current = temp_expansions.pop()
        expanded.add(current)
        new_expansions = set()
        #partition cliques into compatible and incompatible?
        #TODO NOT NECESSARY TO DO ALL!! ONLY CLIQUES THAT WERE NOT TRIED YET!
        #keep index path for each in temp ([qc[i], c[j], c[k], ....])
        #keep compatibility set for each in temp
        #----------
        #make pairwise comp graph for cs. possible combinations are max cliques.
        #for each qc calculate comp with each c and take subgraph
        unions = [tuple(snp.merge(np.array(current), c, duplicates=snp.DROP))
            for c in cliques]
        [validate_quasi_clique(u, valid, incomp) for u in unions]
        new_expansions = [u for u in unions if valid[u] and len(u) > len(current)] #== len(current)+1
        #append and remove all sub-quasi-cliques
        if len(new_expansions) == 0:
            max_expansions.add(current)
        new_expansions = [e for e in new_expansions if e not in expanded]
        temp_expansions.update(new_expansions)
        #temp_expansions = remove_subarrays(temp_expansions)#<---------------
    print(len(max_expansions), len(expanded), len(valid))
    max_expansions = [np.array(e) for e in max_expansions]
    #max_expansions = remove_subarrays(max_expansions)#<---------------
    #print(len(max_expansions))
    return max_expansions if len(max_expansions) > 1 else []

def valid_combo(c1, c2, incomp):
    combo = snp.merge(c1, c2, duplicates=snp.DROP)
    incomps = np.array([incomp[u] for u in combo])
    return len(sorted_unique(incomps)) == len(incomps)

def quasi_clique_expansions2(quasi_cliques, cliques, incomp):
    c_comps = np.array([[valid_combo(cliques[i], cliques[j], incomp)
        if j > i else 0
        for j in range(len(cliques))] for i in range(len(cliques))])
    c_comp_graph = graph_from_matrix(c_comps)
    max_expansions = set()
    for q in quasi_cliques:
        qcomps = np.array([valid_combo(q, cliques[i], incomp)
            for i in range(len(cliques))])
        if np.any(qcomps):
            combos = list(max_cliques(GraphView(c_comp_graph, vfilt=qcomps > 0)))
            combos = [[cliques[i] for i in c] for c in combos]
            max_expansions.update([tuple(np.unique(np.concatenate([q]+co))) 
                for co in combos])
    return [np.array(m) for m in max_expansions] if len(max_expansions) > 1 else []

#iteratively build all possible combinations of cliques while respecting incompatible nodes
def maximal_quasi_cliques(cliques, incompatible):
    #incomp = {n:i for i,ic in enumerate(incompatible) for n in ic}
    incomp = np.zeros(max([n for ic in incompatible for n in ic])+1)
    for i,ic in enumerate(incompatible):
        for n in ic:
            incomp[n] = i
    max_size = max([len(c) for c in cliques])
    quasi_cliques = []
    #go through groups of cliques by size, start with largest cliques
    for s in range(max_size, 2, -1):
        #print("size", s)
        s_cliques = [c for c in cliques if len(c) == s]
        #print("s", [list(c) for c in s_cliques])
        if len(quasi_cliques) == 0:
            quasi_cliques = s_cliques
        #merge cliques depth-first in all combinations until no longer possible
        new_qcs = quasi_clique_expansions2(quasi_cliques, s_cliques, incomp)
        #new_qcs = [q for qs in new_qcs for q in qs] #flatten
        #append and remove all sub-quasi-cliques
        quasi_cliques = remove_subarrays(quasi_cliques+new_qcs)
        #print("num", len(quasi_cliques))
        #print("q", [list(c) for c in quasi_cliques])
    #print("q", len(quasi_cliques))
    return quasi_cliques

def get_edge_combos2(g):
    out_edges = [g.get_out_edges(v, [g.edge_index]) for v in g.get_vertices()]
    edge_combos = []
    #for each vertex find best combinations of neighbors 
    for v in g.get_vertices():#[49:50]:
        n = sorted(g.get_out_neighbors(v))
        #split into connected segments
        #print(v, len(n))
        combos = []
        if len(n) > 0:
            vs = np.array([v]+list(n))
            adjacents = group_adjacent(vs)
            indexes = {v:np.where(vs == v)[0][0] for v in vs}
            filt = g.new_vertex_property("bool")
            filt.a[vs] = True
            gg = GraphView(g, vfilt=filt)
            degs = gg.get_total_degrees(vs)
            neighs = [gg.get_all_neighbors(v) for v in vs]
            #ndegs = [np.mean(gg.get_total_degrees(gg.get_all_neighbors(v))) for v in vs]
            cliques = [np.sort(c) for c in list(max_cliques(gg))]
            qcs = maximal_quasi_cliques(cliques, adjacents)
            if len(qcs) > 0:
                ratings = []
                edges_ids = []
                for q in qcs:
                    filt = g.new_vertex_property("bool")
                    filt.a[q] = True
                    ggg = GraphView(g, vfilt=filt)
                    ratings.append(ggg.num_edges() / ggg.num_vertices())
                    edges_ids.append(ggg.get_edges([g.edge_index])[:,2])
                max_rating = max(ratings)
                best = [q for i,q in enumerate(qcs) if ratings[i] == max_rating]
                print(v, len(n), max_rating, len(best))
                combos = edges_ids
        edge_combos.append(combos)
        #print([list(c) for c in best])
        
        #print(qcs)
        #print(ndegs)
        # graph_draw(gg, vertex_text=g.vertex_index, output_size=(1000, 1000),
        #     output="estg.pdf")
            
            
            # solutions = [np.array([v])]
            # previous_best = -1
            # best = 0
            # while best > previous_best:
            #     print("sol", solutions, best)
            #     previous_best = best
            #     for i,s in enumerate(solutions):
            #         #TODO maybe connected to most, not all
            #         #of nodes connected to all in s take most connected ones
            #         #ONLY IF NOT IN SAME ADJ AS S
            #         #candidates = snp.kway_intersect(*[neighs[indexes[n]] for n in s])
            #         candidates = np.concatenate([a for a in adjacents
            #             if len(snp.intersect(np.array(a), s)) == 0])
            #         print("c", candidates)
            #         cdegs = degs[np.array([indexes[n] for n in candidates])]
            #         max_deg = np.max(cdegs)
            #         most_conn = candidates[cdegs == max_deg]
            #         print("c", most_conn)
            #         #make solutions for each of them
            #         filt = g.new_vertex_property("bool")
            #         filt.a[np.concatenate([s, [most_conn[0]]])] = True
            #         best = GraphView(g, vfilt=filt).num_edges()
            #         print(best)
            #         if best > previous_best:
            #             solutions[i] = [np.sort(np.concatenate([s, np.array([n])]))
            #                 for n in most_conn]
            #     #flatten and keep unique
            #     solutions = [t for s in solutions for t in s]
            #     solutions = [np.array(s) for s in set(map(tuple, solutions))]
            
        #only keep unique
            
            
            # #collect internal edges of subgraph
            # vertices = [v]+list(c)
            # edges = np.concatenate([out_edges[v] for v in vertices])
            # shared = edges[np.where(np.isin(edges[:,1], vertices))]
            # edge_combos[-1].append(sorted(shared[:,2]))
    return edge_combos

#remove all alignments that are not reinforced by others from a simple a-graph
def clean_up(g, time, seg_index):
    #plot_matrix(np.triu(adjacency_matrix(g)), "results/clean0.png")
    #graph_draw(g, output_size=(1000, 1000), output="results/clean_up0.pdf")
    out_edges = [g.get_out_edges(v, [g.edge_index]) for v in g.get_vertices()]
    edge_combos = get_edge_combos2(g)
    
    #look for combos with highest number of edges and lowest number of segments
    seg_combos = []
    for ec in edge_combos:
        current_sc = []
        if len(ec) > 0:
            max_num_edges = max([len(c) for c in ec])
            largest_combos = [c for c in ec if len(c) == max_num_edges]
            #convert to segment combos
            segs = [np.unique(seg_index.a[lc]) for lc in largest_combos]
            min_num_segs = min([len(s) for s in segs])
            if min_num_segs > 0:
                current_sc = [s for s in segs if len(s) == min_num_segs]
        seg_combos.append(current_sc)
        #print(max_num_edges, min_num_segs)
    
    print([len(s) for s in seg_combos])
    
    incompatible = get_incompatible_segs(g, seg_index, out_edges)
    
    #[print(i, list(e), list(edge_combos[i])) for i,e in enumerate(seg_combos)]
    #iteratively select best segment combination for remaining nodes
    sets, ratings = integrate_segment_combos(seg_combos, incompatible)
    best = list(sets[np.argmax(ratings)])
    #print(len(sets), np.max(ratings), best)
    
    threshold = 0.1 #0.01 works well for first example....
    edges = g.get_edges([seg_index])
    while len(ratings) > 0 and max(ratings) > threshold*g.num_edges():
        involved_edges = edges[np.where(np.isin(edges[:,2], best))]
        involved_vertices = np.unique(np.concatenate(involved_edges[:,:2]))
        remaining_vertices = np.setdiff1d(g.get_vertices(), involved_vertices)
        #print(len(involved_vertices), len(remaining_vertices), len(seg_combos))
        remaining_combos = [seg_combos[v] for v in remaining_vertices]
        
        sets, ratings = integrate_segment_combos(remaining_combos, incompatible)
        if len(ratings) > 0 and max(ratings) > threshold*g.num_edges():
            best = best + list(sets[np.argmax(ratings)])
            #print(len(sets), np.max(ratings), list(sets[np.argmax(ratings)]))
    
    #print(edges[:100])
    reduced = Graph(directed=False)
    reduced.add_vertex(len(g.get_vertices()))
    edges = g.get_edges([seg_index])
    edges = edges[np.where(np.isin(edges[:,2], best))]
    reduced.add_edge_list(edges)
    #print(reduced)
    #plot_matrix(np.triu(adjacency_matrix(reduced)), "results/cleani2.png")
    #graph_draw(reduced, output_size=(1000, 1000), output="results/clean_up1.pdf")
    return reduced

def structure_graph(msa, alignment_graph, mask_threshold=.5):
    msa = [[int(m[1:]) if len(m) > 0 else -1 for m in a] for a in msa]
    matches = alignment_graph.new_vertex_property("int")
    matches.a = np.concatenate(msa)
    num_partitions = np.max(matches.a)+1
    # graph_draw(alignment_graph, output_size=(1000, 1000), vertex_fill_color=matches,
    #     output="results/box_of_rain_bars_c.pdf")
    #create connection matrix
    edge_ps = matches.a[alignment_graph.get_edges()] #partition memberships for edges
    edge_ps = edge_ps[np.where(np.all(edge_ps != -1, axis=1))] #filter out non-partitioned
    conn_matrix = np.zeros((num_partitions, num_partitions), dtype=int)
    np.add.at(conn_matrix, tuple(edge_ps.T), 1)
    np.add.at(conn_matrix, tuple(np.flip(edge_ps, axis=1).T), 1)
    conn_matrix = np.triu(conn_matrix, k=1)
    max_conn = np.max(conn_matrix)
    conn_matrix[np.where(conn_matrix < mask_threshold*max_conn)] = 0
    #create graph
    g = graph_from_matrix(conn_matrix)
    #print('created structure graph', g)
    return g, conn_matrix, matches

def pattern_graph(sequences, pairings, alignments):
    #patterns = 
    return

def component_labels(g):
    labels, hist = label_components(g)
    return labels.a

