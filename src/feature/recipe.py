import pandas, logging, sys, os
from dictionary import dictionary_matcher as dm
from dictionary import dictionary_builder as db
logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger("recipe")

class Recipe():
    def __init__(self, name):
        self.name = name
        self.attributes={}

    def init_attributes(self, empty_attributes:dict):
        self.attributes=empty_attributes

    def update_attribute(self, attribute_key, values:list):
        self.attributes[attribute_key]=values

'''
- fast: if true, then we do chunking and set overlap; otherwise bruteforce string indexof                              
- clean: if true, we check single word matches against long matches and if they are part of the long, we ignore them   
'''
def recipe_attribute_extraction(df:pandas.DataFrame, name_col:int, dictionaries:dict, fast, clean):
    recipes = []
    for index, row in df.iterrows():
        name=row[name_col]
        log.info("Processing {}/{} '{}'".format(index, len(df),name))
        r = Recipe(name)
        matches=dm.match_multiple(name, dictionaries, fast,clean)
        for k, d in dictionaries.items():
            if k in matches.keys():
                match_res = matches[k]
                values = [t[0] for t in match_res]
                r.update_attribute(k, values)
            else:
                r.update_attribute(k, [])
        recipes.append(r)

    return recipes

if __name__ == "__main__":
    datacsv = sys.argv[1]
    dictionaryfolder=sys.argv[2]
    dictionaries={}
    for f in os.listdir(dictionaryfolder):
        fullpath = dictionaryfolder+"/{}".format(f)
        dictionaries[f]=db.load_wordlist(fullpath)

    df = pandas.read_csv(datacsv, encoding='utf8', sep=',')
    recipes=recipe_attribute_extraction(df, 1, dictionaries, False, True)

    print("completed")