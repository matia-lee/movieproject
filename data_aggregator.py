import pandas as pd

from pathlib import Path


dir_path = Path("./embeddings_(with_no_titles)")

master_df = pd.DataFrame()

for file_path in dir_path.iterdir():
    if file_path.is_file() and file_path.suffix == '.csv':  
        print(file_path)
        df = pd.read_csv(file_path)

        master_df = pd.concat([master_df, df])


master_df.to_csv("masterdf-updated-notitle.csv")