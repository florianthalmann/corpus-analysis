from collections import defaultdict
import numpy as np
from nltk import CFG

def to_grammar(sequences, sections):
    prod_str = ""
    #add top-level rules
    end_state = np.max(np.hstack(sequences))+1
    #for now removing -1 (but deal with it later!)
    sequences = [np.append(s[s >= 0], end_state) for s in sequences]
    chains = to_chains(sequences)
    chains = add_states(chains, sequences, end_state)
    for c in chains:
        prod_str += str(c[0])+' -> '+' '.join([str(s) for s in c[1:]])+'\n'
    
    #add sections
    for id,sec in sections.items():
        prod_str += str(id)+' -> '+' '.join([str(s) for s in sec])+'\n'
    grammar = CFG.fromstring(prod_str)
    print(grammar)
    # print(len(grammar.productions()))
    # CFG.remove_unitary_rules(grammar)
    # print(len(grammar.productions()))

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

def add_states(chains, sequences, end_state):
    #add start state
    [c.insert(0, 'S') for c in chains
        if any([initial_subseq(c, s) for s in sequences])]
    #add other states
    next_id = np.max(np.hstack(sequences))+1
    ids = {}
    for c in chains:
        if c[0] != 'S':
            #add new id if none defined yet
            if c[0] not in ids:
                ids[c[0]] = next_id
                next_id += 1
            #update later occurrences with id
            for d in chains:
                if d[-1] == c[0]:
                    d[-1] = ids[c[0]]
            #update id
            c.insert(0, ids[c[0]])
    #remove end state
    return [[s for s in c if s != end_state] for c in chains]

def initial_subseq(s1, s2):
    l = min(len(s1), len(s2))
    return np.array_equal(s1[:l], s2[:l])
