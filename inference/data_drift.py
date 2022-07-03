import json
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

# #set column mapping for Evidently Profile
# data_columns = ColumnMapping()
# data_columns.numerical_features = ['Ia', 'Ib','Ic','Va','Vb','Vc']

# #set reference dates
# reference_dates = ('2022-01-01 00:00:00','2022-01-28 23:00:00')

# #set experiment batches dates
# experiment_batches = [
#     ('2022-02-01 00:00:00','2022-02-28 23:00:00'),
#     ('2022-03-01 00:00:00','2022-03-31 23:00:00'),
#     ('2022-04-01 00:00:00','2022-04-30 23:00:00'),
#     ('2022-05-01 00:00:00','2022-05-31 23:00:00'),  
#     ('2022-06-01 00:00:00','2022-06-30 23:00:00'), 
#     ('2022-07-01 00:00:00','2022-07-31 23:00:00'), 
# ]

#evaluate data drift with Evidently Profile
def detect_dataset_drift(train_data, test_data, column_mapping, confidence=0.95, threshold=0.5, get_ratio=False):
    """
    Returns True if Data Drift is detected, else returns False.
    If get_ratio is True, returns ration of drifted features.
    The Data Drift detection depends on the confidence level and the threshold.
    For each individual feature Data Drift is detected with the selected confidence (default value is 0.95).
    Data Drift for the dataset is detected if share of the drifted features is above the selected threshold (default value is 0.5).
    """
    data_drift_profile = Profile(sections=[DataDriftProfileSection()])
    data_drift_profile.calculate(train_data, test_data, column_mapping=column_mapping)
    report = data_drift_profile.json()
    json_report = json.loads(report)

    drifts = []
    num_features = column_mapping.numerical_features if column_mapping.numerical_features else []
    cat_features = column_mapping.categorical_features if column_mapping.categorical_features else []
    for feature in num_features + cat_features:
        drifts.append(json_report['data_drift']['data']['metrics'][feature]['p_value']) 
        
    n_features = len(drifts)
    n_drifted_features = sum([1 if x<(1. - confidence) else 0 for x in drifts])
    
    if get_ratio:
        return n_drifted_features/n_features
    else:
        return True if n_drifted_features/n_features >= threshold else False

# if __name__ == '__main__':
#     X_train_b, X_test_b, y_train_b, y_test_b = split("/Users/pzhn0857/Documents/evidently+mlflow/electrical_fault/data/detect_dataset.csv")
#     drifts = detect_dataset_drift(X_train_b, 
#                            X_test_b, 
#                            column_mapping=data_columns, 
#                            confidence=0.95,
#                            threshold=0.9)
#     print(drifts)
