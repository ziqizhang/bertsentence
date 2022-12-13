import sys
import pandas as pd

from similarity import semsim, stringsim

if __name__ == "__main__":
    #read the data and prepare the two lists
    in_file=sys.argv[1]
    df = pd.read_csv(in_file, encoding='utf8', sep=',')
    list1 = list(df.iloc[:,1])
    #list2 needs some processing
    list2 = list(df.iloc[:,2])
    list2_postprocess=[]
    for l in list2:
        list2_postprocess.append(l.split("(")[1][0:-1])

    semsim_calculator = semsim.Semsim('distiluse-base-multilingual-cased-v1')
    stringsim_calculator = stringsim.StringSim("")
    embeddings1 = semsim_calculator.encode(list1)
    embeddings2 = semsim_calculator.encode(list2)

    for i in range(0, len(list1)):
        semsim_score=semsim_calculator.similarity([embeddings1[i]],[embeddings2[i]])
        stringsim_score=stringsim_calculator.similarity(list1[i], list2[i])
        print("{},{},{},{}".format(list1[i], list2_postprocess[i], semsim_score[0][0], stringsim_score))



