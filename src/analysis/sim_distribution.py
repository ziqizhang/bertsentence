'''
run string sim and sem sim on an input csv and produce distribution of their score range in 10 bins
'''
import sys, logging
import pandas as pd
import matplotlib.pyplot as plt

from similarity import semsim, stringsim
logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger("sim_distribution")

if __name__ == "__main__":
    #read the data and prepare the two lists
    in_file=sys.argv[1]
    df = pd.read_csv(in_file, encoding='utf8', sep=',')
    list1 = list(df.iloc[:,1])
    #list2 needs some processing
    list2 = list(df.iloc[:,2])
    labels=list(df.iloc[:,3])
    list2_postprocess=[]
    for l in list2:
        list2_postprocess.append(l.split("(")[1][0:-1])

    semsim_calculator = semsim.Semsim('distiluse-base-multilingual-cased-v1')
    stringsim_calculator = stringsim.StringSim("")
    embeddings1 = semsim_calculator.encode(list1)
    embeddings2 = semsim_calculator.encode(list2)

    sim_scores=[]
    for i in range(0, len(list1)):
        semsim_score=semsim_calculator.similarity([embeddings1[i]],[embeddings2[i]])
        stringsim_score=stringsim_calculator.similarity(list1[i], list2[i])
        sim_scores.append([list1[i], list2_postprocess[i], semsim_score[0][0], stringsim_score, labels[i]])

    log.info("Plotting distributions")
    df = pd.DataFrame(sim_scores, columns=['Meituan', 'Meishijie','BertSim','Lev', 'Label'])
    bertsim_pos = list(df[df['Label'] == 1].iloc[:,2])
    bertsim_neg = list(df[df['Label'] == 0].iloc[:,2])

    lev_pos = list(df[df['Label'] == 1].iloc[:,3])
    lev_neg = list(df[df['Label'] == 0].iloc[:, 3])
    _bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    plt.hist(bertsim_pos, bins=_bins)
    #plt.show()
    plt.savefig('bertsim_pos.png')
    plt.clf()
    plt.hist(bertsim_neg, bins=_bins)
    # plt.show()
    plt.savefig('bertsim_neg.png')
    plt.clf()

    plt.hist(lev_pos, bins=_bins)
    #plt.show()
    plt.savefig('lev_pos.png')
    plt.clf()
    plt.hist(lev_neg, bins=_bins)
    # plt.show()
    plt.savefig('lev_neg.png')
    plt.clf()

    log.info("Completed")