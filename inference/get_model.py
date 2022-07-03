import os
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from minio import Minio
import pandas as pd
import pprint
from data_evaluate import process
from get_credentials import get_credentials_with_ldap
import mlflow

MINIO_SERVER = os.environ.get("miniohost", 'localhost:9000')

def get_model(run_id):
     # server = "localhost:9000"
    username = "a1"
    password = "123456"
    objectname = "/" + run_id + "/artifacts/model/model.pkl"

    credentials = get_credentials_with_ldap(MINIO_SERVER, username, password, 1800)
    # print("--------------------------------")
    # pprint.pprint(credentials)
    # print("--------------------------------")

    client = Minio(
        MINIO_SERVER,
        secure=False,
        access_key=credentials['AccessKeyId'],
        secret_key=credentials['SecretAccessKey'],
        session_token=credentials['SessionToken']
    )

    # buckets = client.list_buckets()
    # for bucket in buckets:
    #     print(bucket.name, bucket.creation_date)

    # bucket_name = "ningmou-test"
    file = client.get_object(
        bucket_name= "mlflow",
        object_name= objectname
    )
    
    model = pd.read_pickle(file)
    return model