#!/usr/bin/env python
import warnings
from data_drift import detect_dataset_drift
from data_split import split
from get_performance import model_performance
from training import train
from evidently.pipeline.column_mapping import ColumnMapping

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    data_columns = ColumnMapping()
    data_columns.numerical_features = ['Ia', 'Ib','Ic','Va','Vb','Vc']
    #1/2 as old data
    X_train_b, X_test_b, y_train_b, y_test_b = split("data/detect_dataset1.csv")
    run_id = train(X_train_b, X_test_b, y_train_b, y_test_b)
    #1/2 as new labeled data
    X_train_b, X_test_b, y_train_b, y_test_b = split("data/detect_dataset2.csv")
    cnt = 0
    acc = 0
    while (acc < 0.9) and cnt < 5:
        run_id = train(X_train_b, X_test_b, y_train_b, y_test_b)
        acc = model_performance("data/detect_dataset2.csv", run_id)
        cnt = cnt + 1  

    # if acc > 0.9:
    #     serve model 
    # else:
    #     alert 
    if acc < 0.9:
        print("Alert: Cannot find proper hyperparameter.")
    