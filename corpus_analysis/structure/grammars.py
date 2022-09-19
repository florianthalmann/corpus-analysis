from math import log2
from collections import defaultdict
import numpy as np
from nltk import CFG, PCFG, induce_pcfg, Nonterminal, Tree, InsideChartParser,\
    EarleyChartParser, IncrementalTopDownChartParser, Production
#from nltk.parse import pchart
from .hierarchies import to_hierarchy
from ..util import multiprocess, flatten

def to_grammar(sequences, sections):
    end_state = np.max(np.hstack(sequences))+1
    #for now removing -1 (but deal with it later!)
    sequences = [np.append(s[s >= 0], end_state) for s in sequences]
    new_seqs = to_productions(sequences, end_state)
    trees = [Tree.fromstring(to_tree(s[1:], sections, s[0])) for s in new_seqs]
    prods = [p for t in trees for p in t.productions()]
    prods = induce_pcfg(Nonterminal('S'), prods).productions()
    grammar_string = '\n'.join([str(p) for p in prods])
    for k in set([s[0] for s in new_seqs if s[0] != 'S']):
        grammar_string = grammar_string.replace("'"+str(k)+"'", str(k))
    grammar = PCFG.fromstring(grammar_string)
    print(grammar)
    parser = InsideChartParser(grammar)
    #parser.trace(1)
    sentences = [Tree.fromstring(to_tree(s[:-1], sections)).leaves() for s in sequences]
    parses = flatten(multiprocess('parsing', parser.parse_all, sentences), 1)
    probs = mean_probs(parses, grammar)
    print(probs)

def to_pcfg(sequences, sections):
    sequences = [s[s >= 0] for s in sequences]
    trees = [Tree.fromstring(to_tree(s, sections)) for s in sequences]
    # [t.collapse_unary(collapsePOS = False) for t in trees]
    # [t.chomsky_normal_form(horzMarkov = 2) for t in trees]
    prods = [p for t in trees for p in t.productions()]
    return induce_pcfg(Nonterminal('S'), prods)

def pcfg_from_tree(tree):
    #print(to_string_tree(tree))
    prods = Tree.fromstring(to_string_tree(tree)).productions()
    return induce_pcfg(Nonterminal(tree[0]), prods)

def to_string_tree(tree):
    if len(tree) == 1: return str(tree[0])
    return '('+' '.join((str(tree[0]), *[to_string_tree(t) for t in tree[1:]]))+')'

def to_cfg(sequences, sections):
    prod_str = ""
    #add top-level rules
    end_state = np.max(np.hstack(sequences))+1
    #for now removing -1 (but deal with it later!)
    sequences = [np.append(s[s >= 0], end_state) for s in sequences]
    for p in to_productions(sequences, end_state):
        prod_str += str(p[0])+' -> '+' '.join([str(s) for s in p[1:]])+'\n'
    #add sections
    for id,sec in sections.items():
        prod_str += str(id)+' -> '+' '.join([str(s) for s in sec])+'\n'
    grammar = CFG.fromstring(prod_str)
    print(grammar)
    # print(len(grammar.productions()))
    # CFG.remove_unitary_rules(grammar)
    # print(len(grammar.productions()))

def to_productions(sequences, end_state):
    prods = to_chains(sequences)
    #add start state
    [p.insert(0, 'S') for p in prods
        if any([initial_subseq(p, s) for s in sequences])]
    #add other states
    next_id = np.max(np.hstack(sequences))+1
    ids = {}
    for p in prods:
        if p[0] != 'S':
            #add new id if none defined yet
            if p[0] not in ids:
                ids[p[0]] = next_id
                next_id += 1
            #update later occurrences with id
            for q in prods:
                if q[-1] == p[0]:
                    q[-1] = ids[p[0]]
            #update id
            p.insert(0, ids[p[0]])
    #remove end state
    return [[s for s in p if s != end_state] for p in prods]

def initial_subseq(s1, s2):
    l = min(len(s1), len(s2))
    return np.array_equal(s1[:l], s2[:l])

#finds all unique chains in the given sequences
def to_chains(sequences):
    successors = defaultdict(set)
    for s in sequences:
        for i,j in zip(s[:-1], s[1:]):
            successors[i].add(j)
    preds = np.bincount(list(successors.keys()))
    succs = np.bincount([e for v in successors.values() for e in v])
    chains = []
    for i,j in successors.items():
        c = next((c for c in chains if i in c), None)
        if c and len(j) == 1 and succs[i] == 1:
            c.append(next(iter(j)))
        else:
            [chains.append([i,jj]) for jj in j]
    return chains

def to_tree(sequence, sections, designator='S'):
    return '('+str(designator)+' '+' '.join([
        to_tree(sections[s], sections, s) if s in sections
        else str(s) for s in sequence])+')'

def mean_probs(parsed, pcfg):
    prods = [p.productions() for p in parsed]
    probs = to_prob_map(pcfg.productions())
    #return [np.mean([probs[str(p)] for p in ps]) for ps in prods]
    return [np.sum([-np.log2(probs[str(p)]) for p in ps]) for ps in prods]

def to_prob_map(productions):
    map = {}
    for p in productions:
        pstr, prob = str(p).split('[')
        map[pstr.strip()] = float(prob.split(']')[0])
    return map

def description_length(pcfg, sequences):
    N = len(np.unique([str(r) for p in pcfg.productions() for r in p.rhs()]))
    logN = log2(N)
    pcfg_dl = sum([(1+len(p.rhs()))*logN for p in pcfg.productions()])
    #print('pcfg_dl', pcfg_dl)
    #most probable parse for each sequence
    sequences = [[str(v) for v in s] for s in sequences]
    #print(sequences)
    print('.', end='', flush=True)
    parses = [InsideChartParser(pcfg, beam_size=1000).parse_all(s) for s in sequences]
    #print(parses)
    if len(parses[0]) == 0:
        print(pcfg, sequences, parses)
    parses = [sorted(p, key=lambda t: t.prob(), reverse=True)[0] for p in parses]
    #seq_dl = sum([(len(r.rhs()))*logN for p in parses for r in p.productions()])
    prod = log2(len(pcfg.productions()))
    seq_dl = sum([prod for p in parses for r in p.productions()])
    #print('seq_dl', seq_dl)
    return pcfg_dl + seq_dl

def pcfg_dl(pcfg):
    return sum([(len(p.rhs())) for p in pcfg.productions()])

def creations(pcfg):
    
    return 

def seq_prob(pcfg, sequences):
    parses = [InsideChartParser(pcfg).parse_all(s) for s in sequences]
    print(parses)
    probs = [sum([t.prob() for t in p]) for p in parses]
    print(probs, np.mean(probs))
    # parsed = [InsideChartParser(grammar).parse_all(example[:i]) for i in range(len(example))]
    parsed = [IncrementalTopDownChartParser(pcfg).parse(sequences[0][0:i+1]) for i in range(len(sequences[0]))]
    for i,p in enumerate(parsed):
        print(i)
        for t in p:
            print(t)
    # print(sum([t.prob() for t in parsed]))

def test_pcfg():
    #trees = ['(S (0 1 2) (0 2 3))', '(S (0 1 2) (0 4 5))', '(S (1 1 2) (1 2 3))']
    #trees = ['(S (0 1 2) (3 4 5 6))', '(S (0 1 2) 8 (3 4 5 6))', '(S (0 1 2) 4 6)']
    #trees = ['(S (0 1 2) (3 4 5 6))', '(S (0 1 2) (7 8) (3 4 5 6))', '(S (0 1 2) (3 4 6))']
    trees = ['(S (0 1 2 3) (0 1 2 3))', '(S (0 1 2 3) (0 1 2 3))', '(S (0 1 2 3) 4 (0 1 2 3))', '(S (0 1 2 3) 1 3)']
    #trees = ['(S (0 1 2 3) (0 1 2 3))', '(S (0 1 2 3) (0 1 2 3))', '(S (0 1 2 3) 4 (0 1 2 3))', '(S (0 1 2 3) (0 1 3))']
    trees = [Tree.fromstring(t) for t in trees]
    print(trees)
    prods = [p for t in trees for p in t.productions()]
    print(prods)
    grammar = induce_pcfg(Nonterminal('S'), prods)
    print(grammar)
    sequences = [t.leaves() for t in trees]
    print('SEQS', sequences)
    #description_length(grammar, sequences)
    seq_prob(grammar, sequences)

#test_pcfg()