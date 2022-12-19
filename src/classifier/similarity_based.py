#uses bert similarity or a string similarity, using a threshold
from similarity import semsim, stringsim
from classifier import result_summariser_dmresults
import util, sys

def classify(pairs, threshold, method, model_path):
    res=[]
    if method=='bert':
        list1=[]
        list2=[]
        for p in pairs:
            list1.append(p[0])
            list2.append(p[1])
        calc = semsim.Semsim(model_path)
        embeddings1 = calc.encode(list1)
        embeddings2 = calc.encode(list2)
        for i in range(0, len(pairs)):
            semsim_score = calc.similarity([embeddings1[i]], [embeddings2[i]])
            if semsim_score>=threshold:
                res.append(1)
            else:
                res.append(0)
    else:
        calc=stringsim.StringSim("")
        for p in pairs:
            stringsim_score=calc.similarity(p[0],p[1])
            if stringsim_score>=threshold:
                res.append(1)
            else:
                res.append(0)

    return res

def calculate_prf1(predictions, goldstandard, setting,out_dir):
    tp, tr, tf1 = result_summariser_dmresults.save_scores(predictions, goldstandard,
                                                          setting, 3, out_dir)
    print(
        "\t>> Result for this batch || F1:\t{} | Prec:\t{} | Rec:\t{} || Ex/s: X".format(tf1, tp,tr))
    return tp, tr, tf1

if __name__ == "__main__":
    # read the data and prepare the two lists
    in_file = sys.argv[1]
    data = util.load_recipe_gs(in_file)
    #prepare pairs
    pairs=[]
    goldstandard_labels=[]
    for r in data:
        pairs.append((r[2], r[3]))
        goldstandard_labels.append(r[1])

    thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9]

    results_table=[["threshold","precision","recall","f1"]]
    for t in thresholds:
        print("Experiment with threshold={}".format(t))
        predictions=classify(pairs, t, sys.argv[3], sys.argv[4])
        tp, tr, tf1=calculate_prf1(predictions, goldstandard_labels, "{}_{}".format(sys.argv[3], t), sys.argv[2])
        results_table.append([t, tp, tr, tf1])

    for r in results_table:
        print(r)
    #try every threshold with 0.1 increment
