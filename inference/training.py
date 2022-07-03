#!/usr/bin/env python
from random import random
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
import pandas as pd
from sklearn.model_selection import train_test_split

import pickle
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import precision_recall_fscore_support

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def train(X_train_b, X_test_b, y_train_b, y_test_b):
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
    os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
    os.environ['ARTIFACT_ROOT']= 's3://mlflow'

    acc = 0
    loss = 1

    while acc < 0.9 or loss > 0.1: 
        # enable auto logging
        mlflow.xgboost.autolog()
    
        mlflow.set_tracking_uri(mlflow_tracking_uri)  
        if(mlflow.get_experiment_by_name("exp") is None):
            mlflow.create_experiment("exp", artifact_location='s3://mlflow')
        else:
            mlflow.set_experiment("exp")
        # mlflow.set_experiment("exp")
        with mlflow.start_run() as run:

            # train model
            params = {
                'min_child_weight': [1, 5, 10],
                'gamma': [0.5, 1, 1.5, 2, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': [3, 4, 5]
            }
            model = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)
#             model = xgb.train(dtrain, evals=[(dtrain, "train")])
            folds = 10
            param_comb = 5

            skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

            random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train_b, y_train_b), verbose=3, random_state=1001 )

            random_search.fit(X_train_b, y_train_b)
            
            print('\n All results:')
            print(random_search.cv_results_)
            print('\n Best estimator:')
            print(random_search.best_estimator_)
            print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
            print(random_search.best_score_ * 2 - 1)
            print('\n Best hyperparameters:')
            print(random_search.best_params_)

            # evaluate model
            y_proba = random_search.predict_proba(X_test_b)
            y_pred = y_proba.argmax(axis=1)
            loss = log_loss(y_test_b, y_proba)
            acc = accuracy_score(y_test_b, y_pred)
            precision, recall, fscore, support = precision_recall_fscore_support(y_test_b, y_pred, average='macro')            
            mse = mean_squared_error(y_test_b, y_pred)

            # log metrics
            mlflow.log_metrics({"logloss": loss, "accuracy": acc, "precision": precision, 
                                "recall": recall, "fscore": fscore, "mse": mse})
            mlflow.sklearn.log_model(random_search.best_estimator_, "model")
            # filename = 'model.pkl'
            # pickle.dump(random_search.best_estimator_, open(filename, 'wb'))
            run_id = run.info.run_id
        mlflow.end_run()
    return run_id
# 
#     runs = mlflow.search_runs(experiment_ids=experiment_id)
#     runs.head(10)
            
#     runs = mlflow.search_runs(experiment_ids=experiment_id,
#                               order_by=['metrics.mae'], max_results=1)
#             runs.loc[0]
            
#             sleep = True
#             time.sleep(5)
#             sleep = False
