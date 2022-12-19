import pandas as pd
def load_recipe_gs(in_file_csv_path):
    df = pd.read_csv(in_file_csv_path, encoding='utf8', sep=',')
    header = ['id', 'label', 'left_name', 'right_name']

    data = []
    for index, row in df.iterrows():
        if '（' in row[1]:
            left_name = row[1][0: row[1].index('（')].strip()
        elif '(' in row[1]:
            left_name = row[1][0: row[1].index('(')].strip()
        else:
            left_name = row[1]
        data.append([index + 1, row[3], left_name, row[2].split("(")[1][0:-1]])

    return data
