from aeon.classification.convolution_based import HydraClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              precision_score, recall_score, f1_score, 
                              confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import datetime

import warnings
warnings.filterwarnings('ignore')

class HydraCNNClassifier:
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

        model = HydraClassifier(
            n_kernels = config['n_kernels'],
            n_groups = config['n_groups'],
            random_state=config['random_state'],
            n_jobs=config['n_jobs']
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
        Y_pred_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(Y_test, Y_pred)
        report = classification_report(Y_test, Y_pred, output_dict=True, zero_division=0 )
        print("Accuracy:", accuracy)
        print("\nClassification Report:\n", classification_report(Y_test, Y_pred))
        HydraCNNClassifier.visualization(self, Y_test, Y_pred, Y_pred_proba)
        return accuracy, report
    
    
    def visualization(self, Y_test, Y_pred, Y_pred_proba):
        """
        Visualizes the true and predicted labels for the test dataset.
        
        This method creates a scatter plot that displays the true labels (marked with green asterisks) and the predicted labels (marked with red circles) for the test dataset. The plot is labeled with the x-axis as "Sample Index" and the y-axis as "Label", and the title is "True vs Predicted Labels". The legend distinguishes between the true and predicted labels.
        """
        # Визуализация результатов
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        fig.suptitle("Time Series Classification Results", fontsize=16)

        # Истинные и предсказанные метки
        axes[0, 0].scatter(range(len(Y_test)), Y_test, marker='*', label="True Labels", color="green")
        axes[0, 0].scatter(range(len(Y_pred)), Y_pred, marker='o', facecolors='none', edgecolors='r',
                           label="Predicted Labels", color="red", s=30)
        axes[0, 0].set_xlabel("Sample Index")
        axes[0, 0].set_ylabel("Class Label")
        axes[0, 0].set_title("True vs Predicted Labels")
        axes[0, 0].legend()

        # Метрики производительности
        precision = precision_score(Y_test, Y_pred, average=None)
        recall = recall_score(Y_test, Y_pred, average=None)
        f1 = f1_score(Y_test, Y_pred, average=None)
        metrics_df = pd.DataFrame({"Precision": precision, "Recall": recall, "F1-Score": f1}, index=np.unique(Y_test))
        sns.heatmap(metrics_df, annot=True, cmap="YlGnBu", ax=axes[0, 1])
        axes[0, 1].set_title("Performance Metrics")

        # Матрица ошибок
        cm = confusion_matrix(Y_test, Y_pred)
        sns.heatmap(cm, annot=True, cmap="YlGnBu", ax=axes[1, 0])
        axes[1, 0].set_title("Confusion Matrix")
        axes[1, 0].set_xlabel("Predicted Label")
        axes[1, 0].set_ylabel("True Label")
        
        # ROC кривая и AUC
        n_classes = len(np.unique(Y_test))
        Y_test_bin = label_binarize(Y_test, classes=np.unique(Y_test))

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(Y_test, Y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            axes[1, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1, 1].set_xlim([0.0, 1.0])
            axes[1, 1].set_ylim([0.0, 1.05])
            axes[1, 1].set_xlabel('False Positive Rate')
            axes[1, 1].set_ylabel('True Positive Rate')
            axes[1, 1].set_title('Receiver Operating Characteristic (ROC) Curve')
            axes[1, 1].legend(loc="lower right")
        else:
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(Y_test_bin[:, i], Y_pred_proba[:, i]) # type: ignore
                roc_auc = auc(fpr, tpr)
                axes[1, 1].plot(fpr, tpr, lw=2, label=f'ROC curve of class {i} (AUC = {roc_auc:.2f})')

            axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1, 1].set_xlim([0.0, 1.0])
            axes[1, 1].set_ylim([0.0, 1.05])
            axes[1, 1].set_xlabel('False Positive Rate')
            axes[1, 1].set_ylabel('True Positive Rate')
            axes[1, 1].set_title('Receiver Operating Characteristic (ROC) Curve')
            axes[1, 1].legend(loc="lower right")

        # Сохранение графиков
        
        plt.tight_layout()
    
        plt.savefig("graf\\classification_results_" 
                    + datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
                    + ".png", dpi=500, bbox_inches="tight")    
        ## Визуализация процессов
        #plt.figure(figsize=(10, 6))
        #plt.scatter(range(len(Y_test)), Y_test, marker='*', label="True Labels", color="green")
        #plt.scatter(range(len(Y_pred)), Y_pred, marker='o', facecolors='none', edgecolors='r',
        #             label="Predicted Labels", color="red", s=30)
        #plt.xlabel("Sample Index")
        #plt.ylabel("Label")
        #plt.title("True vs Predicted Labels")
        #plt.legend()
        #plt.show()
        

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
        self = HydraCNNClassifier.data_preprocess(self)
        self = HydraCNNClassifier.train_model(self)
        accuracy, report = HydraCNNClassifier.test_model(self)
        return self.model, accuracy, report