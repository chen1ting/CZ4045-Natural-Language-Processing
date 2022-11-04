import os
import pandas as pd


df_list=[]
for file in os.listdir():
    if file.endswith('.tsv'):
        df_list.append(pd.read_csv(file, sep='\t', index_col=[0]))

merge_df = pd.concat(df_list)
merge_df.to_csv('best_params.tsv', sep='\t', index=False)