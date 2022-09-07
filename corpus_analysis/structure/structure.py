import os, psutil, tqdm, datetime
from itertools import product
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from graph_tool.topology import max_cliques
from ..alignment.affinity import matrix_to_segments, segments_to_matrix,\
    get_affinity_matrix, get_alignment_segments, get_alignment_matrix
from ..stats.util import entropy
from ..alignment.smith_waterman import smith_waterman
from ..alignment.eval import get_msa_entropy
from ..util import plot_matrix, mode, profile, plot_sequences, buffered_run,\
    flatten
from .graphs import alignment_graph, structure_graph, component_labels,\
    adjacency_matrix, graph_from_matrix, clean_up
from .hierarchies import make_segments_hierarchical, to_hierarchy,\
    get_hierarchy_labels, get_hierarchies, find_sections_bottom_up,\
    get_most_salient_labels, contract_sections, get_recurring_subseqs, reindex,\
    get_flat_sections_by_coverage, to_hierarchy_labels, transitive_construction_new
from .pattern_graph import super_alignment_graph2, comps_to_seqs, smooth_sequences,\
    cleanup_comps, get_comp_modes, get_mode_sequences, super_alignment_graph3,\
    realign_gaps, realign_gaps_comps, smooth_sequences_with_features, smooth_advanced
from .similarity import sw_similarity, equality, isect_similarity, multi_jaccard
from .form_diagram import form_diagram
from .grammars import to_grammar
from .lexis import lexis, lexis_sections
from .grammar_reduction import find_similars, parts

def remove_blocks(alignment, shape, min_len):
    matrix = segments_to_matrix(alignment, shape)
    #plot_matrix(matrix, 'blocks0.png')
    w = 5#int(min_len/2)
    blocks = np.array([[np.logical_or(
        np.mean(matrix[i, j:min(shape[1],j+w)]) == 1,
        np.mean(matrix[i, max(0,j-w+1):j+1]) == 1)
        for j in range(shape[1])] for i in range(shape[0])])
    #plot_matrix(blocks, 'blocks1.png')
    #print('initial', len(alignment))
    block_segs = [a for a in alignment if np.mean(blocks[tuple(a.T)]) >= 0.9]
    alignment = [a for a in alignment if np.mean(blocks[tuple(a.T)]) < 0.9]
    #print('no blocks', len(alignment))
    #plot_matrix(segments_to_matrix(alignment, shape), 'blocks0.9.png')
    return alignment, block_segs

#clean up and make transitive and hierarchical
def process_alignment(self_alignment, min_len, min_dist, beta, target, put_blocks_back=False):
    #self_alignment, blocks = remove_blocks(self_alignment, shape, min_len)
    #plot_matrix(segments_to_matrix(self_alignment, shape), 'est2.png')
    
    # g, s, i, a, seg  = alignment_graph([length], [[0, 0]], [self_alignment])
    # g2 = clean_up(g, seg)
    # #profile(lambda: clean_up(g, seg))
    # self_alignment = matrix_to_segments(np.triu(adjacency_matrix(g2)))
    # plot_matrix(segments_to_matrix(self_alignment, shape), 'est3.png')
    hierarchy = make_segments_hierarchical(self_alignment, min_len, min_dist, target, beta=beta)#, 'est4')#, 'yoyy')
    
    if put_blocks_back:
        hierarchy += blocks
        #plot_matrix(segments_to_matrix(hierarchy, shape), 'est5.png')
    return hierarchy

def simple_structure(self_alignment, min_len, min_dist, csf, rprop, sfac, nbexp, target=None, lexis=0.2):
    #if len(self_alignment) > 0:
    #print(sequence)
    #plot_matrix(segments_to_matrix(self_alignment, (len(sequence),len(sequence))), 'est1.png')
    
    #hierarchy = process_alignment(self_alignment, min_len, min_dist, beta, target)
    #hmatrix = make_segments_hierarchical(self_alignment, min_len, min_dist, target, beta=beta)
    hmatrix = transitive_construction_new(target, min_dist, min_len, csf, rprop, sfac, nbexp)
    hsegments = matrix_to_segments(hmatrix)
    
    #connected component labels for each position in sequence
    ag, s, i, a, seg = alignment_graph([target.shape[0]], [[0, 0]], [hsegments])
    comp_labels = component_labels(ag)
    # #replace sequence with most frequent value in sequence for each component
    # comp_values = np.array([mode(sequence[np.where(comp_labels == l)])
    #     for l in range(np.max(comp_labels)+1)])
    # improved_sequence = comp_values[comp_labels]
    labels = get_hierarchy_labels([comp_labels], lexis=lexis)[0]
    #plot_sequences(labels, 'hierarchy62.png')
    return labels, hmatrix#np.append(labels, [improved_sequence], axis=0)
    
    #plot_matrix(hierarchy)
    #return sections
    #return np.array([np.repeat(0, len(target))]), target

def matrix_to_labels(matrix, lexis=True):
    g = graph_from_matrix(matrix)[0]
    comp_labels = component_labels(g)
    return get_hierarchy_labels([comp_labels], lexis=lexis)[0]

def shared_structure(sequences, pairings, alignments, msa, min_len, min_dist):
    print("ag")
    ag, s, i, a, seg = alignment_graph([len(s) for s in sequences], pairings, alignments)
    #g = transitive_closure(g)
    print("sg")
    sg, cm, _ = structure_graph(msa, ag)
    plot_matrix(cm, 's1.png')
    matrix = np.triu(adjacency_matrix(sg), k=1)
    plot_matrix(matrix, 's2.png')
    segments = matrix_to_segments(matrix)
    # print("ag2")
    # ag2, s2, i2, a2, seg2 = alignment_graph([len(matrix)], [(0,0)], [segments])
    # print("cu")
    # #clean up structure graph and make connection matrix hierarchical
    # ag3 = clean_up(ag2, seg2)
    # matrix = adjacency_matrix(ag3)
    # plot_matrix(matrix, 's3.png')
    # segments = matrix_to_segments(matrix)
    # print("h")
    # segments = make_segments_hierarchical(segments, min_len, min_dist, len(matrix))
    segments = process_alignment(len(matrix), segments, min_len, min_dist, beta=.25)
    plot_matrix(segments_to_matrix(segments, matrix.shape), 's4.png')
    #plot_matrix(segments_to_matrix(segments, matrix.shape), 'results/structure2.png')
    sg2 = graph_from_matrix(segments_to_matrix(segments, matrix.shape))[0]
    print("l")
    labels = component_labels(sg2)#component labels for each msa column
    #get msa label for each time point in sequences
    msaseqs = [np.array([int(m[1:]) if len(m) > 0 else -1 for m in a]) for a in msa]
    #replace with component labels
    labseqs = [labels[m] for m in msaseqs]
    [np.put(labseqs[i], np.where(msaseqs[i] == -1), -1) for i in range(len(labseqs))]
    plot_sequences(labseqs.copy(), '-seqpat.o.png')
    print("c")
    comps = labels_to_valuecomps(sequences, labseqs)
    #print(labseqs[0])
    #plot_sequences(labseqs)
    print(sum([len(c) for c in comps]), get_msa_entropy(comps))
    
    #hierarchy = get_hierarchies([labels])[0]
    #plot_matrix(hierarchy)
    return comps

def labels_to_valuecomps(sequences, labeled_sequences):
    comps = []
    allseqs = np.concatenate(sequences)
    alllabseqs = np.concatenate(labeled_sequences)
    for i in range(max(alllabseqs)+1):
        comps.append(allseqs[np.where(alllabseqs == i)])
    return comps

def get_msa_eval(song, sequences, msa):
    msa = [np.array([int(m[1:]) if len(m) > 0 else -1 for m in a]) for a in msa]
    comps = labels_to_valuecomps(sequences, msa)
    print(get_msa_entropy(comps), len(comps), sum([len(c) for c in comps]))
    return get_msa_entropy(comps), len(comps), sum([len(c) for c in comps])

def get_old_eval(song, sequences, pairings, alignments, msa, min_len, min_dist):
    #shared_structure(sequences, pairings, alignments, msa, min_len, min_dist)
    comps = buffered_run(song+'-stg', lambda:
        shared_structure(sequences, pairings, alignments, msa, min_len, min_dist), [])
    return get_msa_entropy(comps), len(comps), sum([len(c) for c in comps])

def get_new_eval(song, sequences, pairings, alignments):
    comps = buffered_run(song+'-sag', lambda:
        super_alignment_graph3(song, sequences, pairings, alignments), [])
    comps = cleanup_comps(comps, sequences, song)
    comps = [c for c in comps if len(c) > 4]
    typeseqs = comps_to_seqs(comps, sequences)
    plot_sequences(typeseqs.copy(), song+'-seqpat...png')
    labels = [[sequences[s[0]][s[1]] for s in c] for c in comps]
    return get_msa_entropy(labels), len(comps), sum([len(c) for c in comps])

# def layered_super_alignment_graph2(song, sequences, pairings, alignments):
#     size = np.sum([len(s) for s in sequences])
#     comp_layers = [super_alignment_graph3(song, sequences, p, a)
#         for (p, a) in zip(pairings, alignments)]
#     comp_conns = [csr_matrix((size, size), dtype='int8') for i in range(len(alignments))]
#     for i,l in enumerate(comp_layers):
#         for j,c in enumerate(l):
#             x,y = 
#             for s in c:
#                 comp_conns[i][s[0],s[1]] += 1
#     comp_conns = np.sum(comp_conns)
#     print(np.max(comp_conns))

def layered_super_alignment_graph(song, sequences, pairings, alignments):
    comp_layers = [super_alignment_graph3(song, sequences, p, a)
        for (p, a) in zip(pairings, alignments)]
    label_layers = [comps_to_seqs(comps, sequences) for comps in comp_layers]
    string_labels = [[str(l) for l in zip(*[ll[i] for ll in label_layers])]
        for i in range(len(sequences))]
    uniq_labels = np.unique(flatten(string_labels))
    print(uniq_labels[:10])
    print(len(uniq_labels))
    array_labels = np.array([np.fromstring(l[1:-1], sep=',') for l in uniq_labels])
    print(array_labels[:10])
    
    min_dist = np.ones((len(uniq_labels), len(uniq_labels))) * np.inf
    pos = {l:i for i,l in enumerate(uniq_labels)}
    for ls in string_labels:
        for i in range(len(ls)):
            for j in range(i+1, len(ls)):
                x,y = pos[ls[i]],pos[ls[j]]
                min_dist[x,y] = min(min_dist[x,y], abs(j-i))
                min_dist[y,x] = min_dist[x,y]
    print(min_dist[:10,:10])
    
    num_blank = [len(np.nonzero(l == -1)[0]) for l in array_labels]
    print(num_blank[:10])
    max_num_blank = np.maximum.outer(num_blank, num_blank)
    print(max_num_blank[:10,:10])
    
    same_or_blank = [np.logical_or(np.logical_or(l[:,None] == -1, l == -1), l[:,None] == l)
        for l in array_labels.T]
    print(same_or_blank[1][:10,:10])
    num_same_or_blank = np.sum(same_or_blank, axis=0)
    print(num_same_or_blank[:10,:10])
    lcount = len(comp_layers)
    eq_matrix = (max_num_blank <= 1) & (num_same_or_blank >= lcount)# & (min_dist >= 10)
    print(eq_matrix[:10,:10])
    
    g, w = graph_from_matrix(eq_matrix)[0]
    eq_labels = component_labels(g)
    #print(eq_labels[:10])
    #counts = {u:c for u,c in zip(*np.unique(eq_labels, return_counts=True))}
    #eq_labels = [l if counts[l] > 1 else -1 for l in eq_labels]
    print(eq_labels[:10])
    
    labels = [np.array([eq_labels[np.where(uniq_labels == l)[0][0]] for l in cl])
        for cl in string_labels]
    plot_sequences(labels.copy(), song+'-seqpat..png')

def new_shared_structure(song, sequences, pairings, alignments):
    #print(psutil.Process(os.getpid()).memory_info())
    # no_graph(song, sequences)
    # return
    plot_sequences(sequences, song+'-seqpat.png')
    
    # seqs = [np.zeros(len(s), dtype=int) for s in sequences]
    # for a,p in zip(alignments, pairings):
    #     for aa in a:
    #         seqs[p[0]][aa[:,0]] += 1
    #         seqs[p[1]][aa[:,1]] += 1
    # plot_sequences(seqs.copy(), song+'-seqpat$..png')
    # return
    
    #comps = super_alignment_graph3(song, sequences, pairings, alignments)
    comps = buffered_run(song+'-sag', lambda:
        super_alignment_graph3(song, sequences, pairings, alignments), [])
    #comps = sorted(comps, key=lambda c: np.mean([s[1] for s in c]))
    comps = sorted(comps, key=lambda c: np.mean([min([o[1] for o in c if o[0] == i])
        for i in np.unique([o[0] for o in c])]))
    labels = [[sequences[s[0]][s[1]] for s in c] for c in comps]
    print(sum([len(c) for c in labels]), get_msa_entropy(labels), get_msa_entropy(labels)/sum([len(c) for c in labels]))
    
    #comps = super_alignment_graph3(song, sequences, pairings, alignments)
    #return
    #sag = [s for s in sag if len(s) > 9]
    #print([len(c) for c in comps])
    typeseqs = comps_to_seqs(comps, sequences)
    plot_sequences(typeseqs.copy(), song+'-seqpat..png')
    
    # typeseqs = realign_gaps_comps(sequences, typeseqs, comps)
    # plot_sequences(typeseqs.copy(), song+'-seqpat.png')
    # return
    
    modes = get_comp_modes(sequences, comps)
    # # print(len(typeseqs[0][typeseqs[0] == -1]))
    # # print(sequences[0], modes)
    # matrix, unsmoothed = get_affinity_matrix(sequences[0], modes, True, 4, 0.2)
    # #plot_matrix(matrix)
    
    #return
    
    # print(len(comps))
    # comps = cleanup_comps(comps, sequences, song)
    # labels = [[sequences[s[0]][s[1]] for s in c] for c in comps]
    # print(sum([len(c) for c in labels]), get_msa_entropy(labels), get_msa_entropy(labels)/sum([len(c) for c in labels]))
    # 
    # #comps = cleanup_comps(comps, sequences, song)
    # comps = [c for c in comps if len(c) > 4]
    # # labels = [[sequences[s[0]][s[1]] for s in c] for c in comps]
    # # print(sum([len(c) for c in labels]), get_msa_entropy(labels), get_msa_entropy(labels)/sum([len(c) for c in labels]))
    # # print([len(c) for c in comps])
    # 
    # typeseqs = comps_to_seqs(comps, sequences)
    # plot_sequences(typeseqs.copy(), song+'-seqpat...png')
    
    
    # # typeseqs = buffered_run(song+'-smo',
    # #     lambda: smooth_sequences2(typeseqs, 0.8, 0.6, 10, 5), [])
    # #typeseqs = smooth_sequences(typeseqs, 0.9, 0.5)
    # avg_len = np.mean([len(s) for s in sequences])
    # typeseqs = buffered_run(song+'-smo',
    #     lambda: smooth_sequences_with_features(typeseqs, sequences, comps, 0, 0), [])#, avg_len/2)
    # #typeseqs = smooth_sequences_with_features(typeseqs, sequences, comps, 0, 0)#, avg_len/2)
    # #typeseqs = smooth_sequences(typeseqs, 0.9, 0.5)#, 10, avg_len/2)
    # plot_sequences(typeseqs.copy(), song+'-seqpat....png')
    # comps = labels_to_valuecomps(sequences, typeseqs)
    # print(sum([len(c) for c in comps]), get_msa_entropy(comps),
    #     get_msa_entropy(comps)/sum([len(c) for c in comps]))
    # # typeseqs = smooth_sequences(typeseqs, 0.7, 0.5)
    # # plot_sequences(typeseqs.copy(), song+'-seqpat.....png')
    
    #return
    
    modeseqs = [np.array([modes[c] for c in s]) for s in typeseqs]
    #plot_sequences(get_mode_sequences(sequences, comps), song+'-modes.png')
    #salient, sections, occs = get_most_salient_labels(typeseqs, ignore=[-1], min_len=4)
    #seqs, sections, occs = find_sections_bottom_up(typeseqs, ignore=[-1])
    #plot_sequences(salient, song+'-sali.png')
    #sections = get_flat_sections_by_coverage(typeseqs, [-1])
    
    # def contained(k1, k2, sections):
    #     s1, s2 = sections[k1], sections[k2]
    #     return (k1 in s2) or (k2 in s1) \
    #         or (s2[0] in sections and s1[0] in sections[s2[0]]) \
    #         or (s1[0] in sections and s2[0] in sections[s1[0]]) \
    #         or (s2[1] in sections and s1[1] in sections[s2[1]]) \
    #         or (s1[1] in sections and s2[1] in sections[s1[1]])
    # 
    # k = list(sections.keys())
    # flats = [flatten(to_hierarchy(np.array([s]), sections)) for s in sections]
    # #flats = [[modes[t] for t in s] for s in flats]
    # lens = [len(f) for f in flats]
    # print(lens)
    # sims = np.zeros((len(flats), len(flats)))
    # for i in range(len(flats)):
    #     for j in range(len(flats)):
    #         if i < j and not contained(k[i], k[j], sections)\
    #                 and abs(lens[i]-lens[j]) == 0:
    #             sims[i][j] = sw_similarity(flats[i], flats[j])
    #             if .9 < sims[i][j] <= 1:
    #                 #print(k[i], k[j], sections[k[i]], sections[k[j]])
    #                 print(to_hierarchy(sections[k[i]], sections))
    #                 print(to_hierarchy(sections[k[j]], sections))
    #                 #print(flats[i], flats[j])
    #                 #print(hey)
    # plot_matrix(sims)
    
    # plot_matrix(segments_to_matrix(get_alignment_segments(typeseqs[3], typeseqs[4], 0, 16, 1, 4, .2), (len(typeseqs[3]), len(typeseqs[4])))
    #     +segments_to_matrix(get_alignment_segments(sequences[3], sequences[4], 0, 16, 1, 4, .2), (len(typeseqs[3]), len(typeseqs[4])))
    #     +segments_to_matrix(get_alignment_segments(modeseqs[3], modeseqs[4], 0, 16, 1, 4, .2), (len(typeseqs[3]), len(typeseqs[4]))))
    # 
    
    typeseqs = buffered_run(song+'-smo',
        lambda: smooth_advanced(typeseqs, sequences, comps, 0), [])
    #typeseqs = smooth_advanced(typeseqs, sequences, comps, 1)
    #typeseqs = smooth_advanced(typeseqs, sequences, comps, 2)
    #typeseqs = smooth_advanced(typeseqs, sequences, comps, 3)
    
    plot_sequences(typeseqs, song+'-seqpat.......png')
    # 
    # smoothed = []
    # for j,t in enumerate(typeseqs):
    #     ali = segments_to_matrix(get_alignment_segments(t, consensus, 0, 16, 1, 10, .4), (len(t), len(consensus)))\
    #         +segments_to_matrix(get_alignment_segments(sequences[j], cmodes, 0, 16, 1, 10, .4), (len(t), len(consensus)))
    #     modes = [mode(consensus[np.where(a >= 1)], strict=True) for a in ali]
    #     smoothed.append([m if m >= 0 or t[i] < 0 else t[i] for i,m in enumerate(modes)])
    # plot_sequences(smoothed, song+'-seqpat........png')
    
    #plot_matrix(segments_to_matrix(get_alignment_segments(typeseqs[0], consensus, 0, 16, 1, 10, .4), (len(typeseqs[9]), len(consensus))))
        #+segments_to_matrix(get_alignment_segments(sequences[9], umodes, 0, 16, 1, 4, .2), (len(typeseqs[9]), len(umodes))))
        #+segments_to_matrix(get_alignment_segments(modeseqs[0], umodes, 0, 16, 1, 4, .2), (len(typeseqs[0]), len(umodes))))

    def part_sim(s1, s2):
        sw = smith_waterman(s1, s2)[0]
        minlen = min(len(s1), len(s2))
        return len(sw) / minlen

    def remove_contained(sections):
        secparts = [parts([s], sections) for s in sections.keys()]
        return {s:p for s,p in sections.items() if not any([s in t for t in secparts])}
    
    def replace_repetitions(sections, seqs):
        sections = sections.copy()
        to_remove = []
        for s in sections.keys():
            if len(np.unique(sections[s])) == 1:
                parents = [t for t in sections if s in sections[t]]
                if len(parents) > 0:
                    for p in parents:
                        sections[p] = np.concatenate([sections[s]
                            if c == s else np.array([c]) for c in sections[p]])
                    seqs = [np.concatenate([sections[s]
                        if c == s else np.array([c]) for c in seq]) for seq in seqs]
                    to_remove.append(s)
                    #print(s, sections[s])
        for s in to_remove:
            del sections[s]
        return sections, seqs
    
    #print(nothing)
    #sections, occs = zip(*get_flat_sections_by_coverage(typeseqs, [-1]))
    #seqs, sections, occs = find_sections_bottom_up(typeseqs, ignore=[-1])
    seqs, sections, occs, core = lexis_sections(typeseqs)
    #print(core)
    
    sections, seqs = replace_repetitions(sections, seqs)
    #flats = sections#remove_contained(sections)
    unpacked = {s:to_hierarchy(np.array([s]), sections) for s in sections}
    flats = {s:np.array(flatten(u)) for s,u in unpacked.items()}
    by_coverage = sorted(sections.keys(), key=lambda s: len(flats[s])*len(occs[s]), reverse=True)
    
    # print(by_coverage)
    # print(sorted([len(flats[s])*len(occs[s]) for s in sections.keys()], reverse=True))
    # print([(s, sections[s]) for s in by_coverage[:26]])
    # print(seqs[:3])
    # print([[sections[q] if q in sections else q for q in s] for s in seqs[:3]])
    #print([(s, unpacked[s]) for s in by_coverage[:10]])
    plot_sequences([flats[i] for i in by_coverage], song+'-seqpat......png')
    #flats = [f for f in flats if len(f) > 2]
    #print(flats)
    flats = [flats[s] for s in by_coverage]
    #plot_matrix(get_alignment_matrix(flats[0], np.concatenate(typeseqs[:3]), 0, 16, 1, 16, .2))
    secs = [(s, sections[s]) for s in by_coverage]
    
    print(sections)
    print(find_similars(seqs[:3], sections))
    
    #seqsp = [np.sort(parts(s, sections, False)) for s in seqs]
    #print(seqsp[6:8], isect_similarity(seqsp[6], seqsp[8]))
    #sims = [[multi_jaccard(x, y) for y in seqsp] for x in seqsp]
    #plot_matrix(sims)
    
    print(nothing)
    #print([unpacked[secs[i][0]] for i,f in enumerate(flats) if sw_similarity(f, flats[1]) >= .5])
    plot_sequences([f for f in flats if sw_similarity(f, flats[1]) >= .5], song+'-seqpat.....max.png')
    plot_sequences([f for f in flats if part_sim(f, flats[1]) >= .9], song+'-seqpat.....min.png')
    plot_sequences(np.hstack(to_hierarchy_labels(seqs, sections)[6:9]), song+'-seqpat.....labs.png')
    #plot_sequences(np.hstack(get_hierarchy_labels(typeseqs, ignore=[-1])[:3]), song+'-seqpat.....labs.png')
    print([len(o) for o in occs.values()])
    
    #plot_sequences(flats, song+'-seqpat......png')
    modes = get_comp_modes(sequences, comps)
    feats = [modes[f] for f in flats.values()]
    equivs = np.zeros((len(flats), len(flats)))
    consensus = np.concatenate(flats)
    #DO IT WITH AFFINITY!!!
    
    # ali = get_alignment_matrix(consensus, consensus, 0, 10, 1, 10, .2)
    # plot_matrix(ali, 'aliss.png')
    
    #print(nothing)
    
    #print(list(sections.keys()), np.hstack(list(sections.values())))
    
    # print(entropy(np.concatenate(sequences)+1),
    #     entropy(np.concatenate(typeseqs)+1),
    #     entropy(np.concatenate(seqs)+1),
    #     entropy(np.concatenate(flats)+1),
    #     entropy(np.array(list(np.hstack(list(sections.values())))+list(sections.keys()), dtype=int)))
    
    for i,j in product(range(len(flats)), range(len(flats))):
        if i < j:
            #equivs[i][j] = np.array_equal(feats[i], feats[j])#sw_similarity(sections[i], sections[j])
            equivs[i][j] = sw_similarity(flats[i], flats[j]) >= .5
            #equivs[i][j] = part_sim(flats[i], flats[j]) > .9# * part_sim(feats[i], feats[j])
            # if equivs[i][j] >= .9:
            #     print(sections[i], sections[j], flats[i], flats[j], feats[i], feats[j])
            #     print(equivs[i][j], sw_similarity(feats[i], feats[j]))
    
    plot_matrix(equivs, 'aliss.png')
    
    g, w = graph_from_matrix(equivs)
    eqclasses = component_labels(g)
    byeq = []
    for i in np.unique(eqclasses):
        for j in np.where(eqclasses == i)[0]:
            byeq.append(flats[j])
        byeq.append(np.array([], dtype=int))
    print(byeq)
    plot_sequences(byeq.copy(), song+'-seqpat.....png')
    # for i in np.unique(eqclasses):
    #     locs = np.where(eqclasses == i)[0]
    #     first = locs[0]
    #     for s in typeseqs:
    #         s[np.where(np.isin(s, locs))] = first
    # typeseqs = reindex(typeseqs)
    # plot_sequences(typeseqs.copy(), song+'-seqpat.....png')
    
    # sims = np.zeros((len(sections), len(sections)))
    # for i,j in product(range(len(sections)), range(len(sections))):
    #     if i < j and len(flats[i]) == len(flats[j]):
    #         sims[i][j] = len(np.where(flats[i] == flats[j])[0])/len(flats[j])#sw_similarity(sections[i], sections[j])
    #         if sims[i][j] > 0:
    #             print(flats[i], flats[j], feats[i], feats[j])
    
    # sims = np.zeros((len(sections), len(sections)))
    # subsections = [ for s in sections]
    # for i,j in product(range(len(sections)), range(len(sections))):
    #     if i < j:
    #         sims[i][j] = len(np.intersect1d(sections[i], sections[j]))#sw_similarity(sections[i], sections[j])
    #         if sims[i][j] > 1:
    #             print(sections[i], sections[j])
    
    #to_grammar(seqs, sections)
    
    # salientseqs = contract_sections(salient, sections, occs)
    # print(salientseqs[:2])
    # [print(i, to_hierarchy(sections[i], sections)) for i in [157, 181, 189, 188]]
    # print(206, to_hierarchy(sections[206], sections))
    # print(sections)
    
    # salient, sections, occs = get_most_salient_labels(salientseqs)
    # plot_sequences(salient, song+'-sali1.png')
    
    # print(salientseqs[:2])
    # flat = [flatten(to_hierarchy(np.array(s), sections)) for s in salientseqs[:2]]
    # print(flat)
    # print([len(f) for f in flat])
    # print(sw_similarity(*flat, [-1]))
    # 
    # subseqs = get_recurring_subseqs(salientseqs)
    # subseqs = [s for s in subseqs if not np.all(np.array(s[0]) == -1)]
    # print(len(subseqs))
    # 
    # return
    # 
    # #print(sections)
    # surfaces = [flatten(to_hierarchy(np.array(s), sections)) for s in subseqs]
    # print(surfaces[:2])
    # surfaces = [modes[s] for s in surfaces]
    # print(surfaces[:2])
    # #print(modes[[flatten(to_hierarchy(np.array(s), sections)) for s in salientseqs]])
    # 
    # #surfaces = surfaces[:5]
    # 
    # #now pairwise alignment!!!
    # matrix = np.zeros((len(surfaces), len(surfaces)))
    # for i,s in enumerate(surfaces):
    #     for j,t in enumerate(surfaces[i+1:], i+1):
    #         if abs(len(s) - len(t)) / min(len(s), len(t)) <= 0.1:
    #             if isect_similarity(s, t) > 0.9:
    #                 matrix[i][j] = sw_similarity(s, t, ignore=[-1])
    #                 if matrix[i][j] > 0.9:
    #                     print(len(s), subseqs[i],
    #                         len(t), subseqs[j],
    #                         sw_similarity(s, t, ignore=[-1]))
    #                     salientseqs
    # plot_matrix(matrix)
    # return
    
    #form_diagram(salientseqs)#, sections)
    
    # hierarchies = get_hierarchy_labels(typeseqs)
    # print(len(hierarchies), hierarchies[0].shape, hierarchies[1].shape)
    # #artificially insert maximum so that plots look nicer....
    # maxx = np.max(np.hstack(hierarchies))
    # for h in hierarchies:
    #     h[-1][-1] = maxx
    # plot_sequences(hierarchies[0], song+'-seqpat.....png')
    # plot_sequences(hierarchies[1], song+'-seqpat......png')
    # plot_sequences(hierarchies[2], song+'-seqpat.......png')
    # plot_sequences(hierarchies[30], song+'-seqpat........png')
    # #print([len(h.T) for h in hierarchies[:10]], [len(h) for h in sequences[:10]])
    # plot_sequences([h[1] for h in hierarchies], song+'-seqpat.........png')
    # plot_sequences([h[12] for h in hierarchies], song+'-seqpat..........png')
    
    #get_msa_entropy(comps)
    return get_msa_entropy(labels_to_valuecomps(sequences, typeseqs)), len(comps), sum([len(c) for c in comps])

def no_graph(song, sequences):
    plot_sequences(sequences.copy(), song+'-seqpat,.png')
    typeseqs = smooth_sequences(sequences, 0.7, 0.5)
    plot_sequences(typeseqs.copy(), song+'-seqpat,,.png')
    hierarchies = get_hierarchy_labels(typeseqs)
    print(len(hierarchies), hierarchies[0].shape, hierarchies[1].shape)
    #artificially insert maximum so that plots look nicer....
    maxx = np.max(np.hstack(hierarchies))
    for h in hierarchies:
        h[-1][-1] = maxx
    plot_sequences(hierarchies[0], song+'-seqpat,,,.png')
    plot_sequences(hierarchies[1], song+'-seqpat,,,,.png')
    plot_sequences(hierarchies[2], song+'-seqpat,,,,,.png')
    #print([len(h.T) for h in hierarchies[:10]], [len(h) for h in sequences[:10]])
    plot_sequences([h[1] for h in hierarchies], song+'-seqpat,,,,,,.png')

#print(split_segment(np.array([[0,1],[1,2],[2,3],[3,4],[4,5]]), np.array([0,2,4,10])))
