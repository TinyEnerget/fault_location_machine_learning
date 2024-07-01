from aeon.classification.interval_based import TimeSeriesForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

import numpy as np

class TimeSeriesClassifierForest:
    """
        A class for training and evaluating a time series classification model using a random forest approach.

        Args:
            config (dict): A dictionary containing the configuration parameters for the model.
            X (numpy.ndarray): The input data.
            Y (numpy.ndarray): The target labels.

        Attributes:
            X (numpy.ndarray): The input data.
            Y (numpy.ndarray): The target labels.
            config (dict): The configuration parameters for the model.
            X_train (numpy.ndarray): The training input data.
            X_test (numpy.ndarray): The testing input data.
            Y_train (numpy.ndarray): The training target labels.
            Y_test (numpy.ndarray): The testing target labels.
            model (TimeSeriesForestClassifier): The trained model.

        Methods:
            data_preprocess(): Preprocesses the input data by splitting it into training and testing sets.
            train_model(): Trains the time series classification model using the provided configuration.
            test_model(): Evaluates the trained model on the testing data and prints the accuracy and classification report.
            main(): Runs the data preprocessing, model training, and model testing steps.
        """
        
    def __init__(self, 
                 config: dict,
                 X: np.ndarray,
                 Y: np.ndarray
                 ):
        self.X = X
        self.Y = Y
        self.config = config


    def data_preprocess(self):
        """
        Preprocesses the input data by splitting it into training and testing sets.
        
        Args:
            X (numpy.ndarray): The input data.
            Y (numpy.ndarray): The target labels.
        
        Returns:
            self: The updated instance of the `TimeSeriesClassifierForest` class.
        """
        X = self.X
        Y = self.Y
        X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                            Y, 
                                                            test_size=0.2,
                                                            random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        return self
    
    def train_model(self):
        """
        Trains the time series classification model using the provided configuration.
        
        Args:
            X_train (numpy.ndarray): The training input data.
            Y_train (numpy.ndarray): The training target labels.
            config (dict): The configuration parameters for the model.
        
        Returns:
            self: The updated instance of the `TimeSeriesClassifierForest` class.
        """
                
        X_train = self.X_train
        Y_train = self.Y_train
        config = self.config

        model = TimeSeriesForestClassifier(
            base_estimator=config['base_estimator'][0],
            n_estimators=config['n_estimators'][0],
            n_intervals=config['n_intervals'][0],
            min_interval_length=config['min_interval_length'][0],
            max_interval_length=config['max_interval_length'][0],
            time_limit_in_minutes=config['time_limit_in_minutes'][0],
            contract_max_n_estimators=config['contract_max_n_estimators'][0],
            random_state=config['random_state'][0],
            n_jobs=config['n_jobs'][0],
            parallel_backend= config['parallel_backend']
        )
        model.fit(X_train, Y_train)
        self.model = model
        return self

    def test_model(self):
        """
        Evaluates the performance of the trained time series classification model on the test dataset.
        
        Args:
            self (TimeSeriesClassifierForest): The instance of the `TimeSeriesClassifierForest` class.
        
        Returns:
            self (TimeSeriesClassifierForest): The updated instance of the `TimeSeriesClassifierForest` class.
        """
                
        X_test = self.X_test
        Y_test = self.Y_test
        model = self.model

        Y_pred = model.predict(X_test)

        accuracy = accuracy_score(Y_test, Y_pred)
        report = classification_report(Y_test, Y_pred, output_dict=True, zero_division=0 )
        print("Accuracy:", accuracy)
        print("\nClassification Report:\n", classification_report(Y_test, Y_pred))

        # Визуализация процессов
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(Y_test)), Y_test, marker='*', label="True Labels", color="green")
        plt.scatter(range(len(Y_pred)), Y_pred, marker='o', facecolors='none', edgecolors='r',
                     label="Predicted Labels", color="red", s=30)
        plt.xlabel("Sample Index")
        plt.ylabel("Label")
        plt.title("True vs Predicted Labels")
        plt.legend()
        plt.show()
        return accuracy, report

    def main(self):
        """
        Executes the main workflow of the time series classification model.
        
        This method performs the following steps:
        1. Preprocess the data using the `data_preprocess()` method.
        2. Train the time series classification model using the `train_model()` method.
        3. Evaluate the performance of the trained model on the test dataset using the `test_model()` method.
        4. Return the trained model.
        
        Returns:
            TimeSeriesForestClassifier: The trained time series classification model.
        """
        self = TimeSeriesClassifierForest.data_preprocess(self)
        self = TimeSeriesClassifierForest.train_model(self)
        accuracy, report = TimeSeriesClassifierForest.test_model(self)
        return self.model, accuracy, report