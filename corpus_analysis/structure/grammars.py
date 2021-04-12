from collections import defaultdict
from nltk import CFG

def to_grammar(sequences, sections):
    prod_str = ""
    #add top-level sequences
    #for now removing -1 (but deal with it later!)
    sequences = [s[s >= 0] for s in sequences]
    starts = set([s[0] for s in sequences])
    prod_str += 'S -> '+' | '.join([str(s) for s in starts])+'\n'
    print(CFG.fromstring(prod_str))
    successors = defaultdict(set)
    for s in sequences:
        for i,j in zip(s[:-1], s[1:]):
            successors[i].add(j)
    for i,j in successors.items():
        prod_str += str(i)+' -> '+' | '.join([str(s) for s in j])+'\n'
    print(CFG.fromstring(prod_str))
    #add sections
    for id,sec in sections.items():
        prod_str += str(id)+' -> '+' '.join([str(s) for s in sec])+'\n'
    grammar = CFG.fromstring(prod_str)
    print(len(grammar.productions()))
    CFG.remove_unitary_rules(grammar)
    print(len(grammar.productions()))