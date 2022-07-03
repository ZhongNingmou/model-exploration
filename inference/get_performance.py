from doctest import master
import pickle
from get_model import get_model
from data_evaluate import process
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

def model_performance(filename,run_id):
    model = get_model(run_id)
    X, y = process(filename)
    # model.fit(X, y)
    #2/10 as evaluation data
    result = model.score(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    precision, recall, fscore, support = precision_recall_fscore_support(y, y_pred, average='macro')            
    mse = mean_squared_error(y, y_pred)
    print("accuracy: %d" % acc)
    print("precision: %f" % precision)
    print("recall: %f" % recall)        
    print("fscore: %f" % fscore)
    print("mse: %f" % mse)
    return acc
   