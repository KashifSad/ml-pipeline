import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml



def load_params(params_path:str) -> float:
    global test_size
    test_size = yaml.safe_load(open(params_path,'r'))['data_ingestion']['test_size']
    return test_size

def read_data(url:str) -> pd.DataFrame:
    df = pd.read_csv(url)
    return df

def process_data(df:pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=['tweet_id'],inplace=True)
    final_df = df[df['sentiment'].isin(["happiness","sadness"])]
    final_df['sentiment'].replace({'happiness':1,'sadness':0},inplace=True)
    return final_df

def save_data(data_path,train_data,test_data):
    os.makedirs(data_path)
    train_data.to_csv(os.path.join(data_path,"train.csv"))
    test_data.to_csv(os.path.join(data_path,"test.csv"))
    


def main():
    test_data = load_params('params.yaml')
    df = read_data("https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv")
    final_df = process_data(df)
    train_data, test_data = train_test_split(final_df,test_size=test_size,random_state=42)
    data_path = os.path.join("data","raw")
    save_data(data_path,train_data,test_data)



if __name__ == '__main__':
    main()
