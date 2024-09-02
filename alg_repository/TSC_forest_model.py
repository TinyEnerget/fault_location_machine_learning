from aeon.classification.interval_based import TimeSeriesForestClassifier
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
import os
import re

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
        marker = np.array(range(len(Y)))
        X_train, X_test, Y_train, Y_test, marker_train, marker_test = train_test_split(X, 
                                                            Y,
                                                            marker,
                                                            test_size=0.2,
                                                            random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.marker_test = marker_test
        #pd.DataFrame(Y_train).to_csv('Y_train.csv')
        #pd.DataFrame(marker_train).to_csv('marker_train.csv')
        #pd.DataFrame(marker_test).to_csv('marker_test.csv')

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
            base_estimator=config['base_estimator'],
            n_estimators=config['n_estimators'],
            n_intervals=config['n_intervals'],
            min_interval_length=config['min_interval_length'],
            max_interval_length=config['max_interval_length'],
            time_limit_in_minutes=config['time_limit_in_minutes'],
            contract_max_n_estimators=config['contract_max_n_estimators'],
            random_state=config['random_state'],
            n_jobs=config['n_jobs'],
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
        Y_pred_proba = model.predict_proba(X_test)
        print("Y_pred:", Y_pred_proba)
        pd.DataFrame(Y_test).to_csv('Y_test.csv')
        pd.DataFrame(Y_pred).to_csv('Y_pred.csv')
        pd.DataFrame(Y_pred_proba).to_csv('Y_pred_proba.csv')
        accuracy = accuracy_score(Y_test, Y_pred)
        report = classification_report(Y_test, Y_pred, output_dict=True, zero_division=0 )
        print("Accuracy:", accuracy)
        print("\nClassification Report:\n", classification_report(Y_test, Y_pred))
        self.visualization(Y_test, Y_pred, Y_pred_proba)
        self.test_model_efficiency(Y_pred, Y_pred_proba, self.marker_test)

        return accuracy, report
    
    
    def visualization(self, Y_test, Y_pred, Y_pred_proba):
        """
        Visualizes the true and predicted labels for the test dataset.
        
        This method creates a scatter plot that displays the true labels (marked with green asterisks) and the predicted labels (marked with red circles) for the test dataset. The plot is labeled with the x-axis as "Sample Index" and the y-axis as "Label", and the title is "True vs Predicted Labels". The legend distinguishes between the true and predicted labels.
        """
        # Визуализация результатов
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30, 20))
        #fig.suptitle("Time Series Classification Results", fontsize=30)

        # Истинные и предсказанные метки
        axes[0, 0].scatter(Y_test, Y_pred, marker='o', color="black", 
                           facecolors='black', edgecolors='black', s=30)
        #axes[0, 0].scatter(range(len(Y_pred)), Y_pred, marker='o', facecolors='none', edgecolors='r',
        #                   label="Predicted Labels", color="red", s=30)
        axes[0, 0].plot(Y_test, Y_test, color="red", linewidth=3)
        axes[0, 0].set_xlabel("Target classes", fontsize=30)
        axes[0, 0].set_ylabel("Predict classes", fontsize=30)
        axes[0, 0].set_yticks(np.unique(Y_pred))
        axes[0, 0].set_xticks(np.unique(Y_test))
        axes[0, 0].set_yticklabels(np.unique(Y_pred), fontsize=30)
        axes[0, 0].set_xticklabels(np.unique(Y_test), fontsize=30)
        axes[0, 0].set_xlim(min(np.unique(Y_test)) - 0.5, max(np.unique(Y_test)) + 0.5)
        axes[0, 0].set_ylim(min(np.unique(Y_test)) - 0.5, max(np.unique(Y_test)) + 0.5)
        axes[0, 0].set_title("True vs Predicted Labels", fontsize=30)
        axes[0, 0].legend()

        # Метрики производительности
        precision = precision_score(Y_test, Y_pred, average=None)
        recall = recall_score(Y_test, Y_pred, average=None)
        f1 = f1_score(Y_test, Y_pred, average=None)
        metrics_df = pd.DataFrame({"Precision": precision, "Recall": recall, "F1-Score": f1}, index=np.unique(Y_pred))
        sns.heatmap(metrics_df, annot=True, cmap="YlGnBu", ax=axes[0, 1],
                    annot_kws={"size": 20})
        axes[0, 1].set_title("Performance Metrics", fontsize=25)
        axes[0, 1].set_xlabel("Metrics", fontsize=25)
        axes[0, 1].set_ylabel("Classes", fontsize=25)
        axes[0, 1].tick_params(axis='both', which='major', labelsize=20)
        cbar = axes[0, 1].collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        
        # Матрица ошибок
        cm = confusion_matrix(Y_test, Y_pred, labels=np.unique(Y_test))
        sns.heatmap(cm, annot=True, cmap="YlGnBu", ax=axes[1, 0],
                    annot_kws={"size": 20})
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
    
        plt.savefig("graf\\classification_results_" 
                    + datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
                    + ".png", dpi=500, bbox_inches="tight")    
        
    def test_model_efficiency(self, 
                              Y_pred: np.ndarray, 
                              Y_pred_proba: np.ndarray,
                              marker_test: np.ndarray):
        fileNameAimReposytory = os.listdir(
            'C:\\Users\\Vlad Titov\\Desktop\\Work\\fault_location_machine_learning\\aim methods'
        )

        fileNameAimReposytoryList = re.findall(
            r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', 
           fileNameAimReposytory[0].split('.')[0])
        
        if 'One' in fileNameAimReposytoryList:

            bins_base = [0, 2.5, np.inf]
            bins_hight = [0, 0.5, 1, 2.5, 5, 10, np.inf]

            columns_name = {
                                "0": "1",
                                "1": "2",
                                "2": "3",
                                "3": "4",
                                "4": "5",
                                "5": "6",
                                "6": "7",
                                "7": "8",
                                "8": "9",
                                "9": "10",
                                "10": "11"
                                }
            
            ErrorsPerMethod = pd.read_csv('Errors per method\\OneSideMethodsErrors.csv')
            ErrorsPerMethod = ErrorsPerMethod.rename(columns=columns_name)
            
            methods_errors_one = self.one_method(
                np.array(ErrorsPerMethod),
                marker_test,
                Y_pred)

            ErrorsPerMethod = np.array(ErrorsPerMethod.drop(['2', '5'], axis=1))
            
            methods_errors = self.methods_for_test(
                ErrorsPerMethod,
                marker_test,
                [1,2,4])

            Result_method_3 = methods_errors[:, 0]
            Result_method_4 = methods_errors[:, 1]
            Result_method_7 = methods_errors[:, 2]

            methods_errors_STO = self.STO_method(
                ErrorsPerMethod, 
                marker_test)

            methods_errors_weight_avg = self.weigth_avg_method(
                ErrorsPerMethod, 
                marker_test, 
                Y_pred_proba)
            
            meta_result = {
                "Result_method_3": Result_method_3,
                "Result_method_4": Result_method_4,
                "Result_method_7": Result_method_7,
                "methods_errors_STO": methods_errors_STO,
                "methods_errors_weight_avg": methods_errors_weight_avg,
                "methods_errors_one": methods_errors_one
            }

        elif "Two" in fileNameAimReposytoryList:

            bins_base = [0, 1, np.inf]
            bins_hight = [0, 0.25, 0.5, 1, 2.5, 5, np.inf]

            columns_name = {
                "0": "1",
                "1": "2",
                "2": "3",
                "3": "4",
                "4": "5",
                "5": "6",
                "6": "7",
                "7": "8",
                "8": "9",
                "9": "10",
                "10": "11",
                "11": "12",
                "12": "13",
                "13": "14",
                "14": "15",
                "15": "16",
                "16": "17",
                "17": "18"
                }

            ErrorsPerMethod = pd.read_csv("Errors per method\\TwoSideMethodsErrors.csv")
            ErrorsPerMethod = ErrorsPerMethod.rename(columns=columns_name)

            Y_pred_prob_name = {
                        "0": "1",
                        "1": "2",
                        "2": "3",
                        "3": "4",
                        "4": "5",
                        "5": "6",
                        "6": "7",
                        "7": "8",
                        "8": "9",
                        "9": "10",
                        "10": "11",
                        "11": "13",
                        "12": "14",
                        "13": "15",
                        "14": "16",
                        "15": "17",
                        "16": "18"
                        }
            
            Y_pred_proba = np.delete(Y_pred_proba, 4, axis=1)
            #Y_pred_proba = Y_pred_prob_name.rename(columns=Y_pred_prob_name).drop('5', axis=1)
            

            methods_errors_one = self.one_method(
                np.array(ErrorsPerMethod), marker_test, Y_pred
            )

            methods_errors = self.methods_for_test(
                np.array(ErrorsPerMethod),
                marker_test,
                [6,10,16])

            Result_method_7 = methods_errors[:, 0]
            Result_method_11 = methods_errors[:, 1]
            Result_method_17 = methods_errors[:, 2]
            
            ErrorsPerMethod = np.array(ErrorsPerMethod.drop(["6", "12"], axis=1))
            
            methods_errors_STO = self.STO_method(
                ErrorsPerMethod, 
                marker_test)

            methods_errors_weight_avg = self.weigth_avg_method(
                ErrorsPerMethod, 
                marker_test, 
                Y_pred_proba)
            
            meta_result = {
                "Result_method_7": Result_method_7,
                "Result_method_11": Result_method_11,
                "Result_method_17": Result_method_17,
                "methods_errors_STO": methods_errors_STO,
                "methods_errors_weight_avg": methods_errors_weight_avg,
                "methods_errors_one": methods_errors_one
            }
            

        self.visualization_effiency(meta_result, bins_base)
        self.visualization_effiency(meta_result, bins_hight)

        return meta_result

    def visualization_effiency(self, meta_result, bins):

        def create_dataframe(data, column_name='error'):
            return pd.DataFrame(data, columns=[column_name])

        def calculate_histogram(df, bins):
            return np.histogram(df.error, bins=bins)

        def plot_histogram(data_dict, categories, figsize=(12, 8)):
            fig, ax = plt.subplots(figsize=figsize, layout='constrained')
            bar_width = 0.8 / len(data_dict)
            x = np.arange(len(categories))

            for i, (label, data) in enumerate(data_dict.items()):
                offset = (i - len(data_dict)/2 + 0.5) * bar_width
                bars = ax.bar(x + offset, data, bar_width, label=label)
                autolabel(ax, bars)

            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.set_yticks(np.arange(0, 1000, 100))
            ax.set_xlabel('Погрешность выбранных методов, %')
            ax.set_ylabel('Количество элементов входящих в промежуток')
            ax.grid(True, linestyle="-", color="0.75")
            fig.legend(loc='outside upper right')
            plt.show()

        def autolabel(ax, rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom')

        def process_and_plot_results(result_dict, bins, categories):
            data_dict = {}
            for label, result in result_dict.items():
                df = create_dataframe(result)
                counts, _ = calculate_histogram(df, bins)
                data_dict[label] = counts

            plot_histogram(data_dict, categories)

        def autocategoies(bins):
            categories = []
            for i in range(len(bins) - 1):
                if bins[i+1] != np.inf:
                    categories.append(f"({bins[i]},{bins[i+1]}]")
                else:
                    categories.append(f"({bins[-2]}, Ꝏ)")
            return categories

        process_and_plot_results(meta_result,bins,autocategoies(bins))

    def weigth_avg_method(
                    self, 
                    ErrorsPerMethod: np.ndarray, 
                    marker_test: np.ndarray, 
                    Y_pred_proba: np.ndarray):

        methods_errors = []

        for idx in range(len(Y_pred_proba)):
            exp = marker_test[idx][0]
            methods_errors.append(np.abs(
                np.mean(
                    ErrorsPerMethod[exp] * Y_pred_proba[idx])))
        
        return np.array(methods_errors)
    
    def one_method(
            self,
            ErrorsPerMethod: np.ndarray,
            marker_test: np.ndarray,
            Y_pred: np.ndarray):

        methods_errors = []

        for idx in range(len(Y_pred)):
            method = Y_pred[idx][0] - 1
            exp = marker_test[idx][0]
            methods_errors.append(np.abs(ErrorsPerMethod[exp][method]))

        return np.array(methods_errors)
            
    def STO_method(
            self,
            ErrorsPerMethod: np.ndarray,
            marker_test: np.ndarray
            ):
        methods_errors = []

        for idx in range(len(marker_test)):
            exp = marker_test[idx][0]
            methods_errors.append(np.abs(np.mean(ErrorsPerMethod[exp])))

        return np.array(methods_errors)

    def methods_for_test(
                        self,
                        ErrorsPerMethod: np.ndarray,
                        marker_test: np.ndarray,
                        MethodsList: list):
        
        method_errors_df = pd.DataFrame(columns=MethodsList)

        for method in MethodsList:
            method_errors = []
            for idx in range(len(marker_test)):
                exp = marker_test[idx][0]
                method_errors.append(
                    np.abs(np.mean(ErrorsPerMethod[exp][method]))
                )
            method_errors_df[method] = method_errors
        
        return np.array(method_errors_df)
    
    

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