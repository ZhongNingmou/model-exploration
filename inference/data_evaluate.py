import pandas as pd

def process(filename):
    # filename = "/Users/pzhn0857/Documents/evidently+mlflow/electrical_fault/data/detect_dataset.csv"
    df_detect = pd.read_csv(filename)
    df_detect = df_detect.drop(columns=["Unnamed: 7", "Unnamed: 8"])
    features = ['Ia', 'Ib','Ic','Va','Vb','Vc']

    detection_data_X = df_detect[features]
    detection_data_Y = df_detect['Output (S)']

    return detection_data_X, detection_data_Y

    