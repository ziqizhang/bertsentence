import sys

import pandas as pd
import csv
def recipe_to_deepmatcher_format(in_file_recipe_csv, out_file):
    df = pd.read_csv(in_file_recipe_csv, encoding='utf8', sep=',')
    header = ['id','label','left_name', 'right_name']

    data=[]
    for index, row in df.iterrows():
        if '（' in row[1]:
            left_name=row[1][0: row[1].index('（')].strip()
        elif '(' in row[1]:
            left_name=row[1][0: row[1].index('(')].strip()
        else:
            left_name=row[1]
        data.append([index+1, row[3], left_name, row[2].split("(")[1][0:-1]])

    with open(out_file, 'w', newline='\n', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(header)
        for r in data:
            writer.writerow(r)

if __name__ == "__main__":
    recipe_to_deepmatcher_format(sys.argv[1], sys.argv[2])


