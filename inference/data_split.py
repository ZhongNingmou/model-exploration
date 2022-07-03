import pandas as pd
from sklearn.model_selection import train_test_split

def split(filename):
    # filename = "/Users/pzhn0857/Documents/evidently+mlflow/electrical_fault/data/detect_dataset.csv"
    df_detect = pd.read_csv(filename)
    df_detect = df_detect.drop(columns=["Unnamed: 7", "Unnamed: 8"])
    features = ['Ia', 'Ib','Ic','Va','Vb','Vc']

    detection_data_X = df_detect[features]
    detection_data_Y = df_detect['Output (S)']

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(detection_data_X,
                                                    detection_data_Y,test_size=0.33,random_state=1)

    return X_train_b, X_test_b, y_train_b, y_test_b

    