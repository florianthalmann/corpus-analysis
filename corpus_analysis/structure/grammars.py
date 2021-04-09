from nltk import CFG

def to_grammar(sequences, sections):
    prod_str = ""
    for id,sec in sections.items():
        prod_str += str(id)+' -> '+' '.join([str(s) for s in sec])+'\n'
    grammar = CFG.fromstring(prod_str)
    print(len(grammar.productions()))
    CFG.remove_unitary_rules(grammar)
    print(len(grammar.productions()))