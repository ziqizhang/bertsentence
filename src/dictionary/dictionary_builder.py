'''
implements methods that processes meishijie and also target dataset to build dictionaries used for the matching
'''
import os

import pandas,sys,re, json, numpy, traceback, hanlp, logging, pickle
logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger("dictionary_builder")
'''
given '粤菜', '浙菜', '苏菜', '沪菜', '豫菜', '西北菜' extract the phrases except '菜'
'''
def extract_location_dictionary(df:pandas.DataFrame, data_col:int, outfile):
    lines=list(df.iloc[:, data_col])
    entries=set()
    for l in lines:
        l=re.sub(r'[^\w\s]','',l)
        for v in l.split("\s+"):
            end=v.index('菜')
            if end!=-1:
                v=v[0:end].strip()
            if len(v)>0:
                entries.add(v)
    ordered = sorted(list(entries))
    f = open(outfile, "w")
    for e in ordered:
        f.write(e+'\n')
    f.close()

def extract_cooking_dictionary(df:pandas.DataFrame, data_col:int, outfile):
    lines=list(df.iloc[:, data_col])
    entries=set()
    for l in lines:
        l=re.sub(r'[^\w\s]','',l)
        for v in l.split("\s+"):
            if len(v)>0:
                entries.add(v)
    ordered = sorted(list(entries))
    f = open(outfile, "w")
    for e in ordered:
        f.write(e+'\n')
    f.close()

def extract_ingrediant_dictionary(df:pandas.DataFrame, data_cols:list, outfile):
    p = re.compile('(?<!\\\\)\'')
    entries = set()
    for c in data_cols:
        lines=list(df.iloc[:, c ])

        for l in lines:
            l = p.sub('\"', l).replace("\\","")
            try:
                obj = json.loads(l)
                for k in obj.keys():
                    if '(' in k:
                        k = k.split("(")[1][0:-1].strip()
                    if '（' in k:
                        k = k.split("（")[1][0:-1].strip()
                    if len(k)>0:
                        entries.add(k.strip())
            except:
                print(l)
                traceback.print_exception(*sys.exc_info())
        ordered = sorted(list(entries))
        f = open(outfile, "w")
        for e in ordered:
            f.write(e+'\n')
    f.close()

def extract_frequency_dictionary(df:pandas.DataFrame, data_col:int, outfolder:str, label:str, stopwords:list, binnumber):
    tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
    cached_dictionary_file=outfolder+'/{}'.format('cached_dictionary.pickle')
    cache_exist = os.path.isfile(cached_dictionary_file)

    if cache_exist:
        log.info("Found cached dictionary built before, loading...")
        with open(cached_dictionary_file, 'rb') as handle:
            freq_dictionary = pickle.load(handle)
    else:
        freq_dictionary={}
        log.info("Cached dictionary not found, rebuilding from data...")
        batch=set()
        batch_size=1000
        log.info("Building dictionary by frequency, total rows={}, batch size={}".format(len(df), batch_size))

        for index, row in df.iterrows():
            text = row[data_col]
            if '(' in text:
                text = text.split("(")[1][0:-1].strip()
            if '（' in text:
                text = text.split("（")[1][0:-1].strip()
            text = re.sub(r'[^\w\s]', ' ', text)
            text = text.replace("\s+"," ").strip()
            batch.add(text)
            if (index%batch_size==0 and index!=0):
                res = tok(list(batch))
                batch.clear()
                log.info("\t>>> updating dictionary, total processed={}/{}".format(index, len(df)))
                update_freq_dictionary(freq_dictionary,res, stopwords)

        if len(batch)>0:
            res=tok(list(batch))
            update_freq_dictionary(freq_dictionary,res, stopwords)
            log.info("\t>>> processing last batch")
        log.info("Completed")
        with open(cached_dictionary_file, 'wb') as handle:
            pickle.dump(freq_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # log.info("Standardising frequency scores...".format(len(freq_dictionary)))
    # max_freq = max(list(freq_dictionary.values()))
    # for k in freq_dictionary.keys():
    #     f = freq_dictionary[k]
    #     norm = f/max_freq
    #     freq_dictionary[k]=norm

    log.info("Re-index dictionary by bin, total dictionary size={}".format(len(freq_dictionary)))
    sorted_freq = sorted(list(freq_dictionary.values()))
    parts=numpy.array_split(sorted_freq, binnumber)
    bins=[]
    for p in parts:
        bins.append(p[-1])
    keys=[]
    values=[]
    for k, v in freq_dictionary.items():
        keys.append(k)
        values.append(v)
    bin_ids=numpy.digitize(values, bins, right=False)
    for bin in range(1,binnumber+1):
        log.info("\t>>> processing bin {}".format(bin))
        selected_dictionary=[]
        for i, x in enumerate(bin_ids):
            if x==bin:
                selected_dictionary.append(keys[i])
        selected_dictionary=sorted(selected_dictionary)
        f = open(outfolder+"/"+label+"_bin"+str(bin), "w")
        for e in selected_dictionary:
            f.write(e + '\n')
        f.close()

def update_freq_dictionary(dictionary:dict, tokens, stopwords):
    for tok_seq in tokens:
        for tok in tok_seq:
            tok=tok.strip()
            nums=re.findall('[0-9]+', tok)
            if tok in stopwords or len(nums)>0:
                continue
            if len(tok)>0:
                if tok in dictionary.keys():
                    dictionary[tok]+=1
                else:
                    dictionary[tok]=1

def load_stopwords(folder):
    stop=set()
    for f in os.listdir(folder):
        with open(folder+"/{}".format(f)) as file:
            lines = file.read().splitlines()
            stop.update(lines)
    return list(stop)


if __name__ == "__main__":
    # tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
    # res=tok(['奥尔良口味香酥脆皮鸡架(秘制)', '晓美焰来到北京立方庭参观自然语义科技公司'])

    '''
    /home/zz/Work/data/recipe_linking/meishijie.csv
1
/home/zz/Work/data/recipe_linking/dictionaries/
meishijie
/home/zz/Work/recipe_matching/stopwords
'''

    '''
    /home/zz/Work/data/recipe_linking/meituan.csv
1
/home/zz/Work/data/recipe_linking/dictionaries/
meituan
/home/zz/Work/recipe_matching/stopwords
'''
    bins=10
    df = pandas.read_csv(sys.argv[1], encoding='utf8', sep=',')
    stopwords=load_stopwords(sys.argv[5])
    extract_frequency_dictionary(df, int(sys.argv[2]), sys.argv[3], sys.argv[4],stopwords,bins)
    exit(0)


    extract_ingrediant_dictionary(df, [10,11],sys.argv[2]+"/dictionary_ingrediant.txt")
    extract_location_dictionary(df, 17, sys.argv[2]+"/dictionary_location.txt")
    extract_cooking_dictionary(df, 4, sys.argv[2]+"/dictionary_cooking.txt")



