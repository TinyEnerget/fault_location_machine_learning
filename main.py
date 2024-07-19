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
    def __init__(
            self
            ):
        """
        Initializes the LearnProcess class with configuration settings and data structures to store the loaded data.
        
        The __init__ method sets up the following attributes:
        - exp_list: A list of feature names from the configuration file.
        - fit_type: The machine learning model type from the configuration file.
        - exp_file_path: The file path for the experiment files from the configuration file.
        - Various DataFrame attributes to store the loaded current, voltage, and angle data.
        - Dictionaries to map measurement types to attribute names.
        - A dictionary to map machine learning model types to their corresponding classes.
        - Calls the _aimData and _load_data methods to initialize the data.
        - Calls the _symmetrical_components method to calculate the symmetrical components of the loaded data.
        """
                
        
        config = json.load(open('config//config.json'))

        self.exp_list = config['feature']

        self.fit_type = config['ml_model_type']

        self.exp_file_path = config['experiment_files_path']

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

        self.measurement_types = {
                'C': ('current_A', 'current_B', 'current_C'),
                'V': ('voltage_A', 'voltage_B', 'voltage_C'),
                'AC': ('current_A_angle', 'current_B_angle', 'current_C_angle'),
                'VC': ('voltage_A_angle', 'voltage_B_angle', 'voltage_C_angle')
                }
        
        self.measurement_types_100fps = {
                'I mag': ('current_A', 'current_B', 'current_C'),
                'U mag': ('voltage_A', 'voltage_B', 'voltage_C'),
                'I angle': ('current_A_angle', 'current_B_angle', 'current_C_angle'),
                'U angle': ('voltage_A_angle', 'voltage_B_angle', 'voltage_C_angle')
                }

        self.symmetrical_components = {
            'current_seq_begin': (
                'current_A_begin',
                'current_B_begin',
                'current_C_begin',
                'current_A_angle_begin',
                'current_B_angle_begin',
                'current_C_angle_begin'),
            'current_seq_end': (
                'current_A_end',
                'current_B_end',
                'current_C_end',
                'current_A_angle_end',
                'current_B_angle_end',
                'current_C_angle_end'),
            'voltage_seq_begin': (
                'voltage_A_begin',
                'voltage_B_begin',
                'voltage_C_begin',
                'voltage_A_angle_begin',
                'voltage_B_angle_begin',
                'voltage_C_angle_begin'),
            'voltage_seq_end': (
                'voltage_A_end',
                'voltage_B_end',
                'voltage_C_end',
                'voltage_A_angle_end',
                'voltage_B_angle_end',
                'voltage_C_angle_end')
        }  

        self.fit_library = {
            'forest': tscf,
            'hydra': hydrc
        }
        
        self.best_method = self._aimData()

        self = self._load_data()

        self = self._symmetrical_components_names()

        self = self._symmetrical_components()



    def _fileclassifier(self,
                        file_path: str,
                        file_names: list[str]
                        ):
        """
        Classifies the files in the specified directory and assigns the data to the corresponding
          attributes of the LearnProcess instance.
        
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

        #This part of code will be fixed in future work

        if file_path == 'CSV PMU 100fps':
            measurement_types = self.measurement_types_100fps
            for name in file_names:     
                word_list = name.split('.')[0].split('_')
                try:
                    meas_type, phase, side = word_list[1][0] + ' ' + word_list[3], word_list[1][1], word_list[2]
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

        elif file_path == "CSV file rep":
            measurement_types = self.measurement_types
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
        file_path = self.exp_file_path
        print(file_path)
        file_names = os.listdir(file_path)
        self = self._fileclassifier(file_path, file_names)

        return self
    
    def _symmetrical_components_names(self):

        self.symmetrical_components_names = []
        for idx in range(len(self.exp_list)):
            if 'seq' in self.exp_list[idx].split('_'):
                self.symmetrical_components_names.append(self.exp_list[idx])

        return self
    def _symmetrical_components(self):
        """
        Calculates the symmetrical components for the given current and voltage measurements.
        
        This method iterates through the keys in the `symmetrical_components` dictionary and calculates the symmetrical components for each set of measurements. The symmetrical components are calculated using the `pmu_sc` function, which takes the current and voltage measurements as input and returns the symmetrical component vectors.
        
        The calculated symmetrical components are then assigned to the corresponding attributes of the `LearnProcess` instance.
        
        Returns:
            LearnProcess: The updated `LearnProcess` instance with the calculated symmetrical components.
        """
        for name in self.symmetrical_components.keys():
            setattr(self, name, pmu_sc(
                    getattr(self, self.symmetrical_components[name][0]),
                    getattr(self, self.symmetrical_components[name][1]),
                    getattr(self, self.symmetrical_components[name][2]),
                    getattr(self, self.symmetrical_components[name][3]),
                    getattr(self, self.symmetrical_components[name][4]),
                    getattr(self, self.symmetrical_components[name][5])
                    ).vectors_calculation())
            print('Completed ', name, '\n')

        for name in self.symmetrical_components_names:
            word_list = name.split('_')
            seq_name = word_list[1]+'_'+word_list[2] if 'angle' not in word_list else word_list[1]+'_'+word_list[2]+'_'+word_list[3]
            setattr(self, name, getattr(self, word_list[0] + '_' + word_list[2] +
                                        '_' + word_list[-1])[seq_name])

        return self

    def concat_data(
            self,
             df: pd.DataFrame, 
             exp_names: list[str]) -> np.ndarray:
         """
         Concatenates the data for each experiment into a single array.
         This method iterates through the experiments and concatenates the data for each experiment into a single array.
         Args:
             df: The data for each experiment.
             exp_names: The names of the experiments.
         Returns:
             np.ndarray: The concatenated data for all experiments.
         """
         return np.array([[df[exp].reset_index(drop=True).values] for exp in exp_names]) 
    
    def _preprocessing(self):  
        """
        Preprocesses the data by concatenating the data for each experiment into a single array.
        
        This method iterates through the experiments and concatenates the data for each experiment into a single array, which is then stored in the `X` attribute of the object.
        
        Args:
            None
        
        Returns:
            self: The current object with the preprocessed data stored in the `X` attribute.
        """
        # Сборка групп данных
        exp_names = ['exp ' + str(indx + 1) for indx in range(3000)]

        meases = self.exp_list
        count = 0
        
        for meas in meases:
                if count == 0:
                    data = self.concat_data(getattr(self, f'{meas}'), exp_names)
                    count += 1
                else:
                    data = np.concatenate(
                        [
                            data,
                            self.concat_data(getattr(self, f'{meas}'), exp_names)

                        ],
                        axis=1
                    )
                    count += 1
        setattr(self, 'X', data)
        
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

