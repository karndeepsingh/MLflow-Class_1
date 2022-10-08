import os
import argparse
from tkinter import E 
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

def get_data():
    url = r"C:\Users\shubham\Documents\winequality-red.csv"

    try:
        df = pd.read_csv(url,sep=";")
        return df   
    except Exception as e:
        raise e
     
def evaluate(y,pred):
    rmse = np.sqrt(mean_squared_error(y,pred))
    mae = mean_absolute_error(y,pred)
    r2 = r2_score(y,pred)

    return rmse, mae, r2

def main(alpha,l1_ratio):
    df = get_data()
    train,test = train_test_split(df,random_state=42)
    train_x = train.drop(["quality"],axis=1)
    test_x = test.drop(["quality"],axis=1)

    train_y = train[["quality"]]
    test_y = test[["quality"]]
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        lr.fit(train_x,train_y)

        pred = lr.predict(test_x)

        rmse,mae,r2 = evaluate(test_y,pred)

        print(f"Elastic net Params: alpha: {alpha}, l1_ratio: {l1_ratio}")
        print(f"Elastic net metric: rmse:{rmse}, mae:{mae},r2:{r2}")

        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1-ratio",l1_ratio)

        mlflow.log_metric("rmse",rmse)
        mlflow.log_metric("mae",mae)
        mlflow.log_metric("r2",r2)

        mlflow.sklearn.log_model(lr, "model-2")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--alpha","-a", type=float, default=0.5)
    args.add_argument("--l1_ratio","-l1", type=float, default=0.5)
    parsed_args = args.parse_args()
    try:
        main(alpha=parsed_args.alpha, l1_ratio=parsed_args.l1_ratio)
    except Exception as e:
        raise e    
        