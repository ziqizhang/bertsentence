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
    sentences = [
        "#中卓炸酱#番茄炒方便面",
        "韩式大冷面",
        "大可乐",
        "飘香牛肉",
        "秘制猪排"
    ]
    embedding = semsim_calculator.encode(sentences)
    print("'#中卓炸酱#番茄炒方便面' similarity with the other four names:")
    print(semsim_calculator.similarity(
        [embedding[0]],
        embedding[1:]
    ))

    print(stringsim_calculator.similarity(
       sentences[0],
        sentences[1:]
    ))

