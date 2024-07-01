from alg_repository.data_importer import DataImporter as di
from alg_repository.TSC_forest_model import TimeSeriesClassifierForest as tscf
import json
import joblib
import numpy as np
import pandas as pd
import datetime
import os

class LearnProcess:

    def __init__(
            self,
            exp_type = None
            ):
        self.current_A_begin = pd.DataFrame()
        self.current_B_begin = pd.DataFrame()
        self.current_C_begin = pd.DataFrame()
        self.current_A_angle_begin = pd.DataFrame()
        self.current_A_angle_end =  pd.DataFrame()
        self.current_B_angle_begin = pd.DataFrame()
        self.current_B_angle_end = pd.DataFrame()
        self.current_C_angle_begin = pd.DataFrame()
        self.current_C_angle_end = pd.DataFrame()
        self.current_A_begin = pd.DataFrame()
        self.current_A_end = pd.DataFrame()
        self.current_B_begin = pd.DataFrame()
        self.current_B_end = pd.DataFrame()
        self.current_C_begin = pd.DataFrame()
        self.current_C_end = pd.DataFrame()
        self.voltage_A_angle_begin = pd.DataFrame()
        self.voltage_A_angle_end = pd.DataFrame()
        self.voltage_B_angle_begin = pd.DataFrame()
        self.voltage_B_angle_end = pd.DataFrame()
        self.voltage_C_angle_begin = pd.DataFrame()
        self.voltage_C_angle_end =pd.DataFrame()
        self.voltage_A_begin = pd.DataFrame()
        self.voltage_A_end = pd.DataFrame()
        self.voltage_B_begin = pd.DataFrame()
        self.voltage_B_end = pd.DataFrame()
        self.voltage_C_begin = pd.DataFrame()
        self.voltage_C_end = pd.DataFrame()
        self.exp_library = [
            'current begin',
            'current end',
            'voltage begin',
            'voltage end',
            'current angle begin',
            'current angle end',
            'voltage angle begin',
            'voltage angle end'
        ]
        self.exp_type = exp_type

    def load_data(self):
        """
        Loads the data for the time series classification model.
        
        This method loads the current and voltage data for the beginning and end of each experiment from CSV files located in the 'CSV file rep' directory. It then renames the columns of the data frames to match the experiment numbers and assigns them to the corresponding attributes of the LearnProcess instance.
        
        Returns:
            LearnProcess: The updated LearnProcess instance with the loaded data.
        """
                
        # Загрузка данных
        file_names = os.listdir('CSV file rep')
        file_path = 'CSV file rep/'
        current_A_angle_begin = di(',',file_path + file_names[0]).main_process()
        current_A_angle_end = di(',',file_path + file_names[1]).main_process()
        current_B_angle_begin = di(',',file_path + file_names[2]).main_process()
        current_B_angle_end = di(',',file_path + file_names[3]).main_process()
        current_C_angle_begin = di(',',file_path + file_names[4]).main_process()
        current_C_angle_end = di(',',file_path + file_names[5]).main_process()
        current_A_begin = di(',',file_path + file_names[6]).main_process()
        current_A_end = di(',',file_path + file_names[7]).main_process()
        current_B_begin = di(',',file_path + file_names[8]).main_process()
        current_B_end = di(',',file_path + file_names[9]).main_process()
        current_C_begin = di(',',file_path + file_names[10]).main_process()
        current_C_end = di(',',file_path + file_names[11]).main_process()
        voltage_A_angle_begin = di(',',file_path + file_names[12]).main_process()
        voltage_A_angle_end = di(',',file_path + file_names[13]).main_process()
        voltage_B_angle_begin = di(',',file_path + file_names[14]).main_process()
        voltage_B_angle_end = di(',',file_path + file_names[15]).main_process()
        voltage_C_angle_begin = di(',',file_path + file_names[16]).main_process()
        voltage_C_angle_end = di(',',file_path + file_names[17]).main_process()
        voltage_A_begin = di(',',file_path + file_names[18]).main_process()
        voltage_A_end = di(',',file_path + file_names[19]).main_process()
        voltage_B_begin = di(',',file_path + file_names[20]).main_process()
        voltage_B_end = di(',',file_path + file_names[21]).main_process()
        voltage_C_begin = di(',',file_path + file_names[22]).main_process()
        voltage_C_end = di(',',file_path + file_names[23]).main_process()
        self.time = di(',',file_path + file_names[24]).main_process()
        #time['Time'] = pd.to_datetime(time['Time'], unit='s', origin='unix').dt.strftime('%S.%f')

        def rename_columns(
                  data_current: pd.DataFrame
                           ) -> pd.DataFrame:
            columns_name = []
            for indx in range(len(data_current.columns)):
                columns_name.append('exp ' + str(indx + 1))

            data_current.columns = columns_name
            return data_current

        self.current_A_begin = rename_columns(current_A_begin)
        self.current_B_begin = rename_columns(current_B_begin)
        self.current_C_begin = rename_columns(current_C_begin)
        self.current_A_angle_begin = rename_columns(current_A_angle_begin)
        self.current_A_angle_end =  rename_columns(current_A_angle_end)
        self.current_B_angle_begin = rename_columns(current_B_angle_begin)
        self.current_B_angle_end = rename_columns(current_B_angle_end)
        self.current_C_angle_begin = rename_columns(current_C_angle_begin)
        self.current_C_angle_end = rename_columns(current_C_angle_end)
        self.current_A_begin = rename_columns(current_A_begin)
        self.current_A_end = rename_columns(current_A_end)
        self.current_B_begin = rename_columns(current_B_begin)
        self.current_B_end = rename_columns(current_B_end)
        self.current_C_begin = rename_columns(current_C_begin)
        self.current_C_end = rename_columns(current_C_end)
        self.voltage_A_angle_begin = rename_columns(voltage_A_angle_begin)
        self.voltage_A_angle_end = rename_columns(voltage_A_angle_end)
        self.voltage_B_angle_begin = rename_columns(voltage_B_angle_begin)
        self.voltage_B_angle_end = rename_columns(voltage_B_angle_end)
        self.voltage_C_angle_begin = rename_columns(voltage_C_angle_begin)
        self.voltage_C_angle_end = rename_columns(voltage_C_angle_end)
        self.voltage_A_begin = rename_columns(voltage_A_begin)
        self.voltage_A_end = rename_columns(voltage_A_end)
        self.voltage_B_begin = rename_columns(voltage_B_begin)
        self.voltage_B_end = rename_columns(voltage_B_end)
        self.voltage_C_begin = rename_columns(voltage_C_begin)
        self.voltage_C_end = rename_columns(voltage_C_end)
        return self
    
    def preprocessing(self):
        """
        Preprocesses the data for the time series classification model.
        
        This method initializes temporary variables to store the current and voltage data for the beginning and end of each experiment. It then iterates through the experiments, extracting the relevant data and storing it in the temporary variables. Finally, it concatenates the data into larger arrays and assigns them to the corresponding attributes of the LearnProcess instance.
        
        Returns:
            LearnProcess: The updated LearnProcess instance with the preprocessed data.
        """
                
        self = LearnProcess.load_data(self)

        # Инициализация временных переменных для выгрузки экспериментов
        data_current_A_beg = []
        data_current_B_beg = []
        data_current_C_beg = []
        data_current_A_end = []
        data_current_B_end = []
        data_current_C_end = []
        data_voltage_A_beg = []
        data_voltage_B_beg = []
        data_voltage_C_beg = []
        data_voltage_A_end = []
        data_voltage_B_end = []
        data_voltage_C_end = []
        data_current_A_beg_ang = []
        data_current_B_beg_ang = []
        data_current_C_beg_ang = []
        data_current_A_end_ang = []
        data_current_B_end_ang = []
        data_current_C_end_ang = []
        data_voltage_A_beg_ang = []
        data_voltage_B_beg_ang = []
        data_voltage_C_beg_ang = []
        data_voltage_A_end_ang = []
        data_voltage_B_end_ang = []
        data_voltage_C_end_ang = []

        # Сборка групп данных
        for indx in range(3000):
            # Подгрузка данных и сбор их в отдельные ячейки по экспериментам
            exp_name = 'exp ' + str(indx + 1)
            data_current_A_beg.append([self.current_A_begin[exp_name].reset_index(drop=True).values])
            data_current_B_beg.append([self.current_B_begin[exp_name].reset_index(drop=True).values])
            data_current_C_beg.append([self.current_C_begin[exp_name].reset_index(drop=True).values])
            data_current_A_end.append([self.current_A_end[exp_name].reset_index(drop=True).values])
            data_current_B_end.append([self.current_B_end[exp_name].reset_index(drop=True).values])
            data_current_C_end.append([self.current_C_end[exp_name].reset_index(drop=True).values])
            data_voltage_A_beg.append([self.voltage_A_begin[exp_name].reset_index(drop=True).values])
            data_voltage_B_beg.append([self.voltage_B_begin[exp_name].reset_index(drop=True).values])
            data_voltage_C_beg.append([self.voltage_C_begin[exp_name].reset_index(drop=True).values])
            data_voltage_A_end.append([self.voltage_A_end[exp_name].reset_index(drop=True).values])
            data_voltage_B_end.append([self.voltage_B_end[exp_name].reset_index(drop=True).values])
            data_voltage_C_end.append([self.voltage_C_end[exp_name].reset_index(drop=True).values])
            data_current_A_beg_ang.append([self.current_A_angle_begin[exp_name].reset_index(drop=True).values])
            data_current_B_beg_ang.append([self.current_B_angle_begin[exp_name].reset_index(drop=True).values])
            data_current_C_beg_ang.append([self.current_C_angle_begin[exp_name].reset_index(drop=True).values])
            data_current_A_end_ang.append([self.current_A_angle_end[exp_name].reset_index(drop=True).values])
            data_current_B_end_ang.append([self.current_B_angle_end[exp_name].reset_index(drop=True).values])
            data_current_C_end_ang.append([self.current_C_angle_end[exp_name].reset_index(drop=True).values])
            data_voltage_A_beg_ang.append([self.voltage_A_angle_begin[exp_name].reset_index(drop=True).values])
            data_voltage_B_beg_ang.append([self.voltage_B_angle_begin[exp_name].reset_index(drop=True).values])
            data_voltage_C_beg_ang.append([self.voltage_C_angle_begin[exp_name].reset_index(drop=True).values])
            data_voltage_A_end_ang.append([self.voltage_A_angle_end[exp_name].reset_index(drop=True).values])
            data_voltage_B_end_ang.append([self.voltage_B_angle_end[exp_name].reset_index(drop=True).values])
            data_voltage_C_end_ang.append([self.voltage_C_angle_end[exp_name].reset_index(drop=True).values])

        # Сборка данных в один список
        self.data_current_beg = np.concatenate((np.array(data_current_A_beg),
                                          np.array(data_current_B_beg),
                                          np.array(data_current_C_beg)), axis = 1)
        self.data_current_end = np.concatenate((np.array(data_current_A_end),
                                          np.array(data_current_B_end),
                                          np.array(data_current_C_end)), axis = 1)
        self.data_voltage_beg = np.concatenate((np.array(data_voltage_A_beg),
                                          np.array(data_voltage_B_beg),
                                          np.array(data_voltage_C_beg)), axis = 1)
        self.data_voltage_end = np.concatenate((np.array(data_voltage_A_end),
                                          np.array(data_voltage_B_end),
                                          np.array(data_voltage_C_end)), axis = 1)
        self.data_current_beg_ang = np.concatenate((np.array(data_current_A_beg_ang),
                                               np.array(data_current_B_beg_ang),
                                               np.array(data_current_C_beg_ang)), axis = 1)
        self.data_current_end_ang = np.concatenate((np.array(data_current_A_end_ang),
                                               np.array(data_current_B_end_ang),
                                               np.array(data_current_C_end_ang)), axis = 1)
        self.data_voltage_beg_ang = np.concatenate((np.array(data_voltage_A_beg_ang),
                                               np.array(data_voltage_B_beg_ang),
                                               np.array(data_voltage_C_beg_ang)), axis = 1)
        self.data_voltage_end_ang = np.concatenate((np.array(data_voltage_A_end_ang),
                                               np.array(data_voltage_B_end_ang),
                                               np.array(data_voltage_C_end_ang)), axis = 1)
        return self
    
    def aimData(self):
        """
        Loads and returns the best method from a file located in the 'method train result' directory.
        
        Returns:
            numpy.ndarray: The best method, transposed.
        """
                
        file_path = 'method train result\\'
        file_names = os.listdir(file_path)
        best_method = di(';',file_path + file_names[0]).main_process()
        best_method = np.array(best_method).transpose().flatten()
        return best_method

    def trainData_one_exp(self):

        self = LearnProcess.preprocessing(self)
   
        variable_name = self.exp_type
        if variable_name == 'current begin':
            self.X = self.data_current_beg
        elif variable_name == 'current end':
            self.X = self.data_current_end
        elif variable_name == 'voltage begin':
            self.X = self.data_voltage_beg
        elif variable_name == 'voltage end':
            self.X = self.data_voltage_end
        elif variable_name == 'current begin angle':
            self.X = self.data_current_beg_ang
        elif variable_name == 'current end angle':
            self.X = self.data_current_end_ang
        elif variable_name == 'voltage begin angle':
            self.X = self.data_voltage_beg_ang
        elif variable_name == 'voltage end angle':
            self.X = self.data_voltage_end_ang

        return self.X

    def trainData_multi_exp(self, exp = None):

        variable_name = exp
        if variable_name == 'current begin':
            self.X = self.data_current_beg
        elif variable_name == 'current end':
            self.X = self.data_current_end
        elif variable_name == 'voltage begin':
            self.X = self.data_voltage_beg
        elif variable_name == 'voltage end':
            self.X = self.data_voltage_end
        elif variable_name == 'current begin angle':
            self.X = self.data_current_beg_ang
        elif variable_name == 'current end angle':
            self.X = self.data_current_end_ang
        elif variable_name == 'voltage begin angle':
            self.X = self.data_voltage_beg_ang
        elif variable_name == 'voltage end angle':
            self.X = self.data_voltage_end_ang
        return self.X

    def configuration(self):
        """
        Configures the parameters for a time series classification model.
        
        Args:
            base_estimator (object, optional): The base estimator to use in the ensemble.
            n_estimators (int, optional): The number of estimators to use in the ensemble.
            n_intervals (str or int, optional): The number of intervals to use for the time series features.
            min_interval_length (int, optional): The minimum length of each interval.
            max_interval_length (float, optional): The maximum length of each interval.
            time_limit_in_minutes (float, optional): The time limit in minutes for the model training.
            contract_max_n_estimators (int, optional): The maximum number of estimators to use in the contracted version of the model.
            random_state (int, optional): The random state to use for reproducibility.
            n_jobs (int, optional): The number of jobs to run in parallel.
            parallel_backend (str, optional): The parallel backend to use for the model training.
        
        Returns:
            dict: A dictionary containing the configured parameters for the time series classification model.
        """
                
        base_estimator=None,
        n_estimators=20,
        n_intervals="sqrt",
        min_interval_length=3,
        max_interval_length=np.inf,
        time_limit_in_minutes=None,
        contract_max_n_estimators=50,
        random_state=None,
        n_jobs=14,
        parallel_backend= 'loky'
        self.config = {
             'base_estimator': base_estimator,
             'n_estimators': n_estimators,
             'n_intervals': n_intervals,
             'min_interval_length': min_interval_length,
             'max_interval_length': max_interval_length,
             'time_limit_in_minutes': time_limit_in_minutes,
             'contract_max_n_estimators': contract_max_n_estimators,
             'random_state': random_state,
             'n_jobs': n_jobs,
             'parallel_backend': parallel_backend
        }
        return self.config
    
    def timenow(self):
        return datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")

    def save_model(self, model, exp_name, acc, report, timenow):
        """
        Saves a trained time series classification model to a file, along with relevant metadata about the model and training process.
        
        Args:
            model (object): The trained time series classification model to be saved.
            exp_name (str): The name of the experiment or model being saved.
            acc (float): The accuracy score of the trained model.
            report (dict): A dictionary containing performance metrics for the trained model.
        
        Returns:
            None
        """
        repository = 'Model\\'
        filepath = 'model' + '_' + timenow + '\\'
        filename = exp_name
        adress = repository + filepath
        self.config['max_interval_length'] = 'infinate'
        if model is not None:
            model_description = {
                'exp_name': exp_name,
                'Date': timenow,
                'config': self.config,
                'model': 'time series forest classification',
                'accuracy': acc,
                'report': report,
                'model address': adress + filename + '.pkl'
            }

            if not os.path.exists(adress):
                os.makedirs(adress)

            with open(adress + filename + '.json', 'w') as f:
                json.dump(model_description, f, indent=4)
                joblib.dump(model, adress + filename + '.pkl')
        return 

    def fit(self, exp = None, timenow = None):
        if exp is not None and timenow is not None and self.exp_type is None:
            print('Training the model: ', exp)
            config = LearnProcess.configuration(self)
            X = LearnProcess.trainData_multi_exp(self, exp)
            Y = LearnProcess.aimData(self)
            model, acc, report = tscf(config, X, Y).main()
            LearnProcess.save_model(self, model, exp, acc, report, timenow)
        else:
            print('Training the model: ', self.exp_type)
            config = LearnProcess.configuration(self)
            X = LearnProcess.trainData_one_exp(self)
            Y = LearnProcess.aimData(self)
            model, acc, report = tscf(config, X, Y).main()
            timenow = LearnProcess.timenow(self)
            LearnProcess.save_model(self, model, self.exp_type, acc, report, timenow)
        return model, acc, report
    
    def FullExpProcess(self):
        self = LearnProcess.preprocessing(self)
        timenow = LearnProcess.timenow(self)
        for exp in self.exp_library:
            LearnProcess.fit(self, exp, timenow)
        return 'Могло быть лучше но так получилось'

