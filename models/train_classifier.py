# import libraries
import sys
import pandas as pd
import numpy as np
import pickle

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import FunctionTransformer,LabelEncoder,OneHotEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data():
    '''
    This funtion is to load the data from database
    
    Argument:

    None

    return:

    X_train: training dataset
    X_test: validate datasest
    y_train: training target dataset
    y_test: validate datasest
    '''
    data = pd.read_csv('../data/data.csv')

    X = data.iloc[:,:-1]
    Y = data.iloc[:,-1]
    seed = 14
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    
    return X_train, X_test, y_train, y_test


def build_model(X_train, X_test, y_train, y_test):
    '''
    Use sklearn pipeline to build up the model
    
    Argument:

    X_train: training dataset
    X_test: validate datasest
    y_train: training target dataset
    y_test: validate datasest
    
    Return:
    
    built model


    '''
    ## Here I will use xgboost classifier for mahcine learning algorithum

    model = XGBClassifier(max_depth=2,min_child_weight=13,learning_rate=0.1,n_estimators=1000,objective='binary:logistic',gamma=0,max_delta_step=0,subsample=1,colsample_bytree=0.6)
    clf = model.fit(X_train,y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, eval_metric='auc', verbose=True)
    y_pred = clf.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(metrics.classification_report(y_test, predictions,target_names = ['class 0', 'class 1']))

    return clf

def optimize_model(X_train,Y_train,X_test,Y_test):
    '''
    Optimize model using GridSearchCV

    Arg:
    model : model object for training
    X_train, Y_train : Training dataset pair
    X_test, Y_test: Evaluating dataset pair
    category_name: category name list

    '''
    model = XGBClassifier(learning_rate=0.5,n_estimators=300,objective='binary:logistic',gamma=0,max_delta_step=0,verbosity=2)
    
    parameters = {
              'max_depth':[2],
              'min_child_weight':[13,15,17],
              'subsample':[1],
              'colsample_bytree':[0.6]
              }

    cv = GridSearchCV(estimator=model, param_grid=parameters,cv=5)
    be = cv.fit(X_train,Y_train.values).best_estimator_
    print(cv.best_params_)
    print(cv.best_score_)
    pred = be.predict(X_test)
    print(classification_report(Y_test.values,pred))

    model = be

def save_model(model, model_filepath):

    '''
    Save trained model to a place using pickle searilizer

    Arg:
    model: trained model
    model_filepath: the saving place for model
    '''

    filename = '{}'.format(model_filepath)
    pickle.dump(model, open(filename, 'wb'))


def main():
    '''
    Run the whole process from loading data to saving the trained model.
    
    '''
    print('Loading data...\n')
    X_train, X_test, Y_train, Y_test = load_data()
    

    #print('Optimizing data...\n')
    #model = optimize_model(X_train,Y_train,X_test,Y_test)
    
    print('Building model...\n')
    model = build_model(X_train, X_test, Y_train, Y_test)
    
        
    print('Saving model...')
    save_model(model,'./classifier.pkl')



if __name__ == '__main__':
    main()