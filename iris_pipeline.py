import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import joblib
import datetime
import logging
import time
from typing import Tuple, Dict
import schedule

# setup logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

class IrisMLPipeline:
    def __init__(self):
        """
        Initialize the ML pipeline with MLflow tracking
        """
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("iris_classification")
        self.model_path="models/iris_model.joblib"
        self.scaler_path="models/scaler.joblib"

    def load_and_preprocess_data(self)->Tuple[np.ndarray,np.ndarray]:
        """
        Load and preprocess iris dataset
        """
        logger.info("Load and preprocess data ...")

        # Load data
        iris=load_iris()
        X,y=iris.data,iris.target

        # Create a dataframe for better data handling
        feature_names=iris.feature_names
        self.df=pd.DataFrame(X,columns=feature_names)

        return X,y


    def train_model(self,X:np.ndarray,y:np.ndarray)->Tuple[RandomForestClassifier,Dict]:
        """
        Train the model and log metrics with MLflow
        """
        logging.info("Training model...")

        # split data
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

        # scaling features
        scaler=StandardScaler()
        X_train_scaled=scaler.fit_transform(X_train)
        X_test_scaled=scaler.transform(X_test)

        # save scaler
        joblib.dump(scaler,self.scaler_path)

        # Define model
        model=RandomForestClassifier(n_estimators=100,random_state=42)

        with mlflow.start_run():
            # Train and evaluate
            model.fit(X_train_scaled,y_train)
            y_pred=model.predict(X_test_scaled)

            # calculate metrics
            accuracy=accuracy_score(y_test,y_pred)
            classification_rep=classification_report(y_test,y_pred)
            
            # log parameters and metrics
            mlflow.log_param("n_estimators",100)
            mlflow.log_metric("accuracy",accuracy)
            mlflow.log_metric("timestamp",time.time())

            # log model
            mlflow.sklearn.log_model(model,"model")

            # save the model locally
            joblib.dump(model,self.model_path)

            metrics={"Accuracy":accuracy,
                     "Classification Report":classification_rep}
            logger.info(f"Traning completed with accuracy: {accuracy}")
            logger.info(f"Classification Report:\n {classification_rep}")

        return model, metrics
    
    def predict(self, X:np.ndarray)->np.ndarray:
        """make predictions using trained model"""
        logger.info("Making predictions...")

        # load the model and scaler
        model=joblib.load(self.model_path)
        scaler=joblib.load(self.scaler_path)

        # preprocess the input
        X_scaled=scaler.transform(X)

        # make predictions
        predictions=model.predict(X_scaled)
        return predictions

    def check_model_performance(self)->bool:
        """ Check if the model needs retraining based on time"""
        try:
            last_train_time=mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name("iris_classification").experiment_id]
                                               )["timestamp"].max()
            # Retrain if the model is older 3min
            return (time.time()-last_train_time)>3*60
        except:
            return True
        
    def retrain_if_needed(self):
        """Check and retrain if needed"""
        logger.info("Checking if retraining is needed...")
        
        if self.check_model_performance():
            logger.info("Retraining the model")
            X,y=self.load_and_preprocess_data()
            self.train_model(X,y)
            logger.info("Retraining completed")
        else:
            logger.info("No retraining needed")


# def main():
# """ Main function to run pipeline """
pipeline=IrisMLPipeline()

# Initial training
X,y=pipeline.load_and_preprocess_data()
pipeline.train_model(X,y)

# Schedule retraining check every 24 hours
schedule.every(3).minutes.do(pipeline.retrain_if_needed)

while True:
    schedule.run_pending()
    time.sleep(60) # sleep for a min between checks


# if __name__=="__main__":
#     # For testing without scheduling
#     pipeline=IrisMLPipeline()
#     X,y=pipeline.load_and_preprocess_data()
#     model,metrics=pipeline.train_model(X,y)

#     # make a test prediction
#     sample_data=X[:1]
#     prediction=pipeline.predict(sample_data)
#     print(f"Sample prediction: {prediction}")






