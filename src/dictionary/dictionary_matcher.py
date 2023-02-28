from dictionary import dictionary_builder as db
import hanlp,logging, sys

logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger("dictionary_matcher")

'''
match a string against multiple dictionaries
- fast: if true, then we do chunking and set overlap; otherwise bruteforce string indexof
- clean: if true, we check single word matches against long matches and if they are part of the long, we ignore them
'''
def match_multiple(string:str, dictionaries:dict, fast:bool, clean:bool):
    res={}
    for k, d in dictionaries.items():
        log.info("\t >> matching against {} items...".format(len(d)))
        if fast:
            matched = match_fast(string, d)
        else:
            matched=match(string, d)

        if clean:
            remove=[]
            for index, item in enumerate(matched):
                if item[1]>1:
                    break
                for x in range(index+1, len(matched)):
                    nextitem=matched[x]
                    if item[1] == nextitem[1]:
                        continue
                    elif item[1]<nextitem[1] and item[0] in nextitem[0]:
                        remove.append(index)
            matched = [i for j, i in enumerate(matched) if j not in remove]

        res[k]=matched
    return res

'''
given a dictionary with a list of strings, search each entry in the target 'string' and return the list of matched
'''
def match(string:str, dictionary:list):
    matched=[]
    string=db.clean_text(string)
    for entry in dictionary:
        if len(entry)>0 and entry in string:
            matched.append((entry, len(entry)))
    matched=sorted(
        matched,
        key=lambda x: x[1]
    )
    return matched

'''
given a dictionary with a list of strings, do a set overlap with the chunks extracted from the target string
'''
def match_fast(string:str, dictionary:set):
    string = db.clean_text(string)
    tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
    tokens=set(tok(string))
    overlap = tokens.intersection(dictionary)
    matched=[]
    for entry in overlap:
        if len(entry)>0:
            matched.append((entry, len(entry)))
    matched=sorted(
        matched,
        key=lambda x: x[1]
    )
    return matched

