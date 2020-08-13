## To deal with files and OS
import os
import sys
import shutil

import joblib

## For date manipulation
import time
from datetime import datetime

## Regular Expression
import re

## Standard data manipulation libraries
import numpy as np
import pandas as pd

## Customized functions
import functions

## For model Estimation
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# Path to data directory.
WRKDIR = "./"
DATADIR = WRKDIR + "ai-workflow-capstone/cs-train/"

## model specific variables (iterate the version and note with each
## change)
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "example random forest on toy data"
SAVED_MODEL = "model-{}.joblib".format(re.sub("\.","_",str(MODEL_VERSION)))

def fetch_data():
    """
    fetch the data for training your model
    """

    df = functions.fetch_data (DATADIR)


    max_countries = df[["country", "price"]].groupby (df["country"]). \
                      sum ().sort_values (by = "price",
                                          ascending = False).index[:10]

    df_max_country = df[df.country.map (lambda x: x in max_countries)]

    # create aggregate data
    df_aggregate = functions.convert_df_to_ts (df, max_countries)

    # create our feature matrix. here mainly lagged variables
    features_mat = functions.engineer_features(df_aggregate,
                                           training = 0)

    return(features_mat)

    ## add test checking that countries are <= 10

def model_train(mode=None):
    """
    example funtion to train model
    
    'mode' -  can be used to subset data essentially simulating a train
    """

    ## data ingestion
    df = fetch_data()

    ## Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(df[0], 
                                                        df[1],
                                                        test_size = 0.3,
                                                        shuffle = False,
                                                        random_state = 1)

    ## preprocessing pipeline
    cat_features = [x for x in X_train.columns if x not in X_train.describe().columns]
    num_features = list(X_train.describe().columns)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)])

    pipe_dtree = Pipeline(steps = 
                    [
                        ('pre', preprocessor),
                        ('dtree', DecisionTreeRegressor(max_depth = 15))
                    ])

    clf = pipe_dtree.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("... saving model: {}".format(SAVED_MODEL))
    joblib.dump(clf,SAVED_MODEL)

def model_load():
    """
    example funtion to load model
    """

    if not os.path.exists(SAVED_MODEL):
        raise Exception("Model '{}' cannot be found did you train the model?".format(SAVED_MODEL))
    
    model = joblib.load(SAVED_MODEL)
    return(model)

def model_predict(query,model=None):
    """
    example funtion to predict from model
    """

    ## load model if needed
    if not model:
        model = model_load()

    query = np.array(query)

    ## output checking
    query = query.reshape(1, -1)

    query = pd.DataFrame(query)

    query.columns = ['previous_7', 'previous_14',
                     'previous_28', 'previous_70',
                     'previous_year', 'recent_invoices',
                     'recent_views']

    ## make prediction and gather data for log entry
    y_pred = model.predict(query)
        
    return(y_pred)

if __name__ == "__main__":

    """
    basic test procedure for model.py
    """
    
    ## train the model
    model_train()

    ## load the model
    model = model_load()

    ex1 = [3.18669700e+04, 5.83556410e+04,
           1.42086181e+05, 3.53433941e+05,
           9.06022210e+04, 6.19666667e+00,
           5.77430000e+02]

    ex2 = [3.89, 5.78,
           7.42086181, 9.42086181,
           2.1904, 6.1966,
           1.7743]

    ## example predict
    for query in [ex1, ex2]:
        y_pred = model_predict(query,model)
        print("predicted: {}".format(y_pred))
