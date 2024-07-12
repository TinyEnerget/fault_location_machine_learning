from alg_repository.data_importer import DataImporter as di
from alg_repository.TSC_forest_model import TimeSeriesClassifierForest as tscf
from alg_repository.hydra_model import HydraCNNClassifier as hydrc
from alg_repository.PMU_symmetrical_components import PMU_symmetrical_components as pmu_sc
import json
import joblib
import numpy as np
import pandas as pd
import datetime
import os

class LearnProcess:
    """
    The `LearnProcess` class is responsible for loading, preprocessing, and training time series classification models. It provides methods to:
    - Load data from CSV files located in the 'CSV file rep' directory
    - Preprocess the loaded data by extracting and concatenating relevant features
    - Configure the parameters for a time series classification model
    - Train a single or multiple experiments using the configured model
    - Save the trained model and relevant metadata to a file
    The class has several attributes that store the preprocessed data, such as `data_current_beg`, `data_current_end`, `data_voltage_beg`, `data_voltage_end`, `data_current_beg_ang`, `data_current_end_ang`, `data_voltage_beg_ang`, and `data_voltage_end_ang`.
    The `fit()` method is the main entry point for training a time series classification model. It takes an optional `exp` parameter to specify the experiment type, and an optional `timenow` parameter to provide the current timestamp. If `exp` and `timenow` are not provided, the method will use the `exp_type` attribute of the `LearnProcess` instance.
    The `FullExpProcess()` method runs the full experiment process, which includes preprocessing the data, getting the current timestamp, and iterating through the `exp_library` list to train a model for each experiment type.
    """
    def __init__(
            self,
            exp_type = None,
            fit_type = 'forest'
            ):
        """
        Initializes the `LearnProcess` class with the specified experiment type and fit type.
        
        The class has several attributes that store the preprocessed data, such as `current_A_begin`, `current_B_begin`, `current_C_begin`, `current_A_angle_begin`, `current_B_angle_begin`, `current_C_angle_begin`, `current_A_end`, `current_B_end`, `current_C_end`, `current_A_angle_end`, `current_B_angle_end`, `current_C_angle_end`, `voltage_A_begin`, `voltage_B_begin`, `voltage_C_begin`, `voltage_A_angle_begin`, `voltage_B_angle_begin`, `voltage_C_angle_begin`, `voltage_A_end`, `voltage_B_end`, `voltage_C_end`, `voltage_A_angle_end`, `voltage_B_angle_end`, and `voltage_C_angle_end`.
        
        The `exp_library` attribute is a list of experiment types, and the `fit_library` attribute is a dictionary that maps fit types to their corresponding classifier classes.
        
        Args:
            exp_type (str, optional): The experiment type. Defaults to `None`.
            fit_type (str, optional): The fit type. Defaults to `'forest'`.
        """
                
        self.current_A_begin = pd.DataFrame()
        self.current_B_begin = pd.DataFrame()
        self.current_C_begin = pd.DataFrame()

        self.current_A_angle_begin = pd.DataFrame()
        self.current_B_angle_begin = pd.DataFrame()
        self.current_C_angle_begin = pd.DataFrame()

        self.current_A_end = pd.DataFrame()
        self.current_B_end = pd.DataFrame()
        self.current_C_end = pd.DataFrame()

        self.current_A_angle_end =  pd.DataFrame()
        self.current_B_angle_end = pd.DataFrame()
        self.current_C_angle_end = pd.DataFrame()

        self.voltage_A_begin = pd.DataFrame()
        self.voltage_B_begin = pd.DataFrame()
        self.voltage_C_begin = pd.DataFrame()  

        self.voltage_A_angle_begin = pd.DataFrame()
        self.voltage_B_angle_begin = pd.DataFrame()
        self.voltage_C_angle_begin =pd.DataFrame()

        self.voltage_A_end = pd.DataFrame()
        self.voltage_B_end = pd.DataFrame()
        self.voltage_C_end = pd.DataFrame()

        self.voltage_A_angle_end = pd.DataFrame()
        self.voltage_B_angle_end = pd.DataFrame()
        self.voltage_C_angle_end =pd.DataFrame()        

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
        self.fit_type = fit_type
        self.fit_library = {
            'forest': tscf,
            'hydra': hydrc
        }
        self.best_method = self._aimData()

        self = self._load_data()


    def _fileclassifier(self,
                        file_path: str,
                        file_names: list
                        ):
        """
        Classifies the files in the specified directory and assigns the data to the corresponding attributes of the LearnProcess instance.
        
        Args:
            file_path (str): The path to the directory containing the CSV files.
            file_names (list): A list of the file names in the directory.
        
        Returns:
            LearnProcess: The updated LearnProcess instance with the loaded data.
        """
                
        def rename_columns(
                  data: pd.DataFrame
                           ) -> pd.DataFrame:
            columns_name = []
            for indx in range(len(data.columns)):
                columns_name.append('exp ' + str(indx + 1))

            data.columns = columns_name
            return data

        measurement_types = {
                'C': ('current_A', 'current_B', 'current_C'),
                'V': ('voltage_A', 'voltage_B', 'voltage_C'),
                'AC': ('current_A_angle', 'current_B_angle', 'current_C_angle'),
                'VC': ('voltage_A_angle', 'voltage_B_angle', 'voltage_C_angle')
                }

        for name in file_names:     
            word_list = name.split('.')[0].split('_')
            try:
                meas_type, phase, side = word_list[2], word_list[3], word_list[4]
            except IndexError:
                self.time = di(',',file_path + '//' + name).main_process()
                continue
            
            if meas_type in measurement_types:
                attr_names = measurement_types[meas_type]
                attr_name = f"{attr_names[ord(phase) - ord('A')]}_{side}"
                print('\n', attr_name, '\n', file_path + '//' + name)
                setattr(self, attr_name,
                         rename_columns(di(',', file_path + '//' + name).main_process()))
                print("Completed!",'\n')
            
        return self

    def _load_data(self):
        """
        Loads the data for the time series classification model.
        This method loads the current and voltage data for the beginning and end of each experiment from
        CSV files located in the 'CSV file rep' directory. It then renames the columns of the data frames
        to match the experiment numbers and assigns them to the corresponding attributes of the LearnProcess instance.
        Returns:
            LearnProcess: The updated LearnProcess instance with the loaded data.
        """  
        # Загрузка данных
        file_path = 'CSV file rep'
        file_names = os.listdir(file_path)
        self = self._fileclassifier(file_path, file_names)

        return self
    
    def _symmetrical_components(self):
        """
        Calculates the symmetrical components of the current and voltage signals.
        
        This method uses the `pmu_sc` function to calculate the positive, negative, and zero sequence components of the current and voltage signals for the beginning and end of each experiment.
        
        The calculated sequence components are stored in the following attributes of the `LearnProcess` instance:
        - `current_seq_begin`: The sequence components of the current signals at the beginning of each experiment.
        - `current_seq_end`: The sequence components of the current signals at the end of each experiment.
        - `voltage_seq_begin`: The sequence components of the voltage signals at the beginning of each experiment.
        - `voltage_seq_end`: The sequence components of the voltage signals at the end of each experiment.
        
        Returns:
            LearnProcess: The updated `LearnProcess` instance with the calculated sequence components.
        """
                
        self.current_seq_begin = pmu_sc(
            self.current_A_begin,
            self.current_B_begin,
            self.current_C_begin,
            self.current_A_angle_begin,
            self.current_B_angle_begin,
            self.current_C_angle_begin
        ).vectors_calculation()
        self.current_seq_end = pmu_sc(
            self.current_A_end,
            self.current_B_end,
            self.current_C_end,
            self.current_A_angle_end,
            self.current_B_angle_end,
            self.current_C_angle_end
        ).vectors_calculation()
        self.voltage_seq_begin = pmu_sc(
            self.voltage_A_begin,
            self.voltage_B_begin,
            self.voltage_C_begin,
            self.voltage_A_angle_begin,
            self.voltage_B_angle_begin,
            self.voltage_C_angle_begin
        ).vectors_calculation()
        self.voltage_seq_end = pmu_sc(
            self.voltage_A_end,
            self.voltage_B_end,
            self.voltage_C_end,
            self.voltage_A_angle_end,
            self.voltage_B_angle_end,
            self.voltage_C_angle_end
        ).vectors_calculation()

        return self

    def _preprocessing(self):
        """
        Preprocesses the data for the time series classification model.
        This method initializes temporary variables to store the current and voltage data for the beginning and end of each experiment.
        It then iterates through the experiments, extracting the relevant data and storing it in the temporary variables.
        Finally, it concatenates the data into larger arrays and assigns them to the corresponding attributes of the LearnProcess instance.
        Returns:
            LearnProcess: The updated LearnProcess instance with the preprocessed data.
        """      
        
        # Сборка групп данных
        exp_names = ['exp ' + str(indx + 1) for indx in range(3000)]
        
        data_current_A_beg = [[self.current_A_begin[exp].reset_index(drop=True).values] for exp in exp_names]
        data_current_B_beg = [[self.current_B_begin[exp].reset_index(drop=True).values] for exp in exp_names]
        data_current_C_beg = [[self.current_C_begin[exp].reset_index(drop=True).values] for exp in exp_names]

        data_current_A_end = [[self.current_A_end[exp].reset_index(drop=True).values] for exp in exp_names]
        data_current_B_end = [[self.current_B_end[exp].reset_index(drop=True).values] for exp in exp_names]
        data_current_C_end = [[self.current_C_end[exp].reset_index(drop=True).values] for exp in exp_names]

        data_voltage_A_beg = [[self.voltage_A_begin[exp].reset_index(drop=True).values] for exp in exp_names]
        data_voltage_B_beg = [[self.voltage_B_begin[exp].reset_index(drop=True).values] for exp in exp_names]
        data_voltage_C_beg = [[self.voltage_C_begin[exp].reset_index(drop=True).values] for exp in exp_names]

        data_voltage_A_end = [[self.voltage_A_end[exp].reset_index(drop=True).values] for exp in exp_names]
        data_voltage_B_end = [[self.voltage_B_end[exp].reset_index(drop=True).values] for exp in exp_names]
        data_voltage_C_end = [[self.voltage_C_end[exp].reset_index(drop=True).values] for exp in exp_names]

        data_current_A_beg_ang = [[self.current_A_angle_begin[exp].reset_index(drop=True).values] for exp in exp_names]
        data_current_B_beg_ang = [[self.current_B_angle_begin[exp].reset_index(drop=True).values] for exp in exp_names]
        data_current_C_beg_ang = [[self.current_C_angle_begin[exp].reset_index(drop=True).values] for exp in exp_names]

        data_current_A_end_ang = [[self.current_A_angle_end[exp].reset_index(drop=True).values] for exp in exp_names]
        data_current_B_end_ang = [[self.current_B_angle_end[exp].reset_index(drop=True).values] for exp in exp_names]
        data_current_C_end_ang = [[self.current_C_angle_end[exp].reset_index(drop=True).values] for exp in exp_names]

        data_voltage_A_beg_ang = [[self.voltage_A_angle_begin[exp].reset_index(drop=True).values] for exp in exp_names]
        data_voltage_B_beg_ang = [[self.voltage_B_angle_begin[exp].reset_index(drop=True).values] for exp in exp_names]
        data_voltage_C_beg_ang = [[self.voltage_C_angle_begin[exp].reset_index(drop=True).values] for exp in exp_names]

        data_voltage_A_end_ang = [[self.voltage_A_angle_end[exp].reset_index(drop=True).values] for exp in exp_names]
        data_voltage_B_end_ang = [[self.voltage_B_angle_end[exp].reset_index(drop=True).values] for exp in exp_names]
        data_voltage_C_end_ang = [[self.voltage_C_angle_end[exp].reset_index(drop=True).values] for exp in exp_names]


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
    
    def _aimData(self):
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

    def _trainData_one_exp(self):
        """
        Trains the data for a single experiment.
        Returns:
            numpy.ndarray: The training data for the specified experiment type.
        """         
        self = LearnProcess._preprocessing(self)
   
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

    def _trainData_multi_exp(self, exp = None):
        """
        Trains the data for multiple experiments.
        Args:
            exp (str, optional): The experiment type to train the data for. Can be one of 'current begin', 
            'current end', 'voltage begin', 'voltage end', 'current begin angle', 'current end angle',
            'voltage begin angle', or 'voltage end angle'.
        Returns:
            numpy.ndarray: The training data for the specified experiment type.
        """

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

    def _configuration(self):
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
            n_kernels (int, optional): The number of kernels to use.
            n_groups (int, optional): The number of groups to use.
        
        Returns:
            dict: A dictionary containing the configured parameters for the time series classification model.
        """
        
        n_kernels = 2,
        n_groups = 4,        
        base_estimator=None,
        n_estimators=20,
        n_intervals="sqrt",
        min_interval_length=3,
        max_interval_length=np.inf,
        time_limit_in_minutes=None,
        contract_max_n_estimators=50,
        random_state=None,
        n_jobs=5,
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
             'parallel_backend': parallel_backend,
             'n_groups': n_groups,
             'n_kernels': n_kernels
        }
        return self.config
    
    def _timenow(self):
        """
        Returns the current date and time as a string in the format "dd_mm_YYYY_HH_MM".
        
        Returns:
            str: The current date and time as a string.
        """
                
        return datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")

    def _save_model(self, model, exp_name, acc, report, timenow):
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
        """
        Trains a time series classification model and saves the trained model along with relevant metadata.
        
        Args:
            exp (str, optional): The name of the experiment or model being trained. If not provided, the `exp_type` attribute of the `LearnProcess` instance will be used.
            timenow (str, optional): The current date and time as a string in the format "dd_mm_YYYY_HH_MM". If not provided, it will be generated using the `_timenow()` method.
        
        Returns:
            tuple: A tuple containing the trained model, the accuracy score, and a report of the model's performance metrics.
        """
                
        if exp is not None and timenow is not None and self.exp_type is None:
            print('Training the model: ', exp)
            config = LearnProcess._configuration(self)
            X = LearnProcess._trainData_multi_exp(self, exp)
            Y = LearnProcess._aimData(self)
            model, acc, report = self.fit_library[self.fit_type](config, X, Y).main()
            LearnProcess._save_model(self, model, exp, acc, report, timenow)
        else:
            print('Training the model: ', self.exp_type)
            config = LearnProcess._configuration(self)
            X = LearnProcess._trainData_one_exp(self)
            Y = LearnProcess._aimData(self)
            model, acc, report = self.fit_library[self.fit_type](config, X, Y).main()
            timenow = LearnProcess._timenow(self)
            LearnProcess._save_model(self, model, self.exp_type, acc, report, timenow)
        return model, acc, report
    
    def FullExpProcess(self):
        """
        Runs the full experiment process for the LearnProcess instance.
        
        This method preprocesses the data, gets the current timestamp, and then iterates through the `exp_library` list, calling the `fit()` method for each experiment.
        
        Returns:
            str: A message indicating the outcome of the full experiment process.
        """
                
        self = LearnProcess._preprocessing(self)
        timenow = LearnProcess._timenow(self)
        for exp in self.exp_library:
            LearnProcess.fit(self, exp, timenow)
        return 'Могло быть лучше, но так получилось'

