import pandas as pd
import numpy as np

from TaxiFareModel.data import *
from TaxiFareModel.encoders import *
from TaxiFareModel.utils import *
from TaxiFareModel.params import BUCKET_NAME
from TaxiFareModel.gcp import *

from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient

import joblib
from google.cloud import storage
from termcolor import colored

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[FR] [PARIS] [MARTIN] trainer + version_1"
MODEL_DIRECTORY = 'Pipeline'


params = dict(
    nrows=10000,
    upload=True,
    local=False,  # set to False to get data from GCP (Storage or BigQuery)
    gridsearch=False,
    optimize=True,
    estimator="xgboost",
    mlflow=True,  # set to True to log params to mlflow
    experiment_name=EXPERIMENT_NAME,
    pipeline_memory=None,  # None if no caching and True if caching expected
    distance_type="manhattan",
    feateng=[
        "distance_to_center", "direction", "distance", "time_features",
        "geohash"
    ],
    n_jobs=-1)  # Try with njobs=1 and njobs = -1


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc',
                               TimeFeaturesEncoder('pickup_datetime')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                         remainder="drop")

        gboost = GradientBoostingRegressor(n_estimators=100)
        ridge = Ridge()
        svm = SVR(C=1, epsilon=0.05)
        adaboost = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=None))

        model = StackingRegressor(
            estimators=[("gboost", gboost),("adaboost", adaboost),("ridge", ridge), ("svm_rbf", svm)],
            final_estimator=LinearRegression(),
            cv=5,
            n_jobs=-1)

        pipe = Pipeline([('preproc', preproc_pipe), ('model', model)],
                        memory="local_path")
        self.pipeline = pipe
        #params
        self.mlflow_log_param('model', 'mixed stacking')

    #def grid_search():
    #    param_grid =  {'kneighborsregressor__n_neighbors': [3,4,5,6,7,8,9,10,15,20,30]}
    #    search_knn = GridSearchCV(self.pipeline, param_grid=param_grid,
    #                          cv=3, n_jobs=-1, verbose=2, scoring=rmse_neg)

    #    search_knn.fit(X, y_log);
    #    print(search_knn.best_params_)
    #    search_knn.best_score_

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X, self.y)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        #metrics
        self.mlflow_log_metric('rmse',rmse)
        return rmse

    def save_model(self):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder"""
        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        joblib.dump(self.pipeline, 'model.joblib')
        print("saved model.joblib locally")

        # Implement here
        storage_upload(MODEL_DIRECTORY, BUCKET_NAME)


    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

if __name__ == "__main__":
    # store the data in a DataFrame
    df = get_data()
    #df = download_data()
    df = clean_data(df)
    #Optimized
    df = df_optimized(df)
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    # build pipeline
    train = Trainer(X_train, y_train)
    train.set_pipeline()
    # train, give type of model
    train.run()
    #...
    experiment_id = train.mlflow_experiment_id
    # evaluate
    rmse = train.evaluate(X_val, y_val)
    print(rmse)
    print(
        f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}"
    )
    train.save_model()
