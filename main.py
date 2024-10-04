import attr
from alg_repository.data_importer import DataImporter as di
from alg_repository.TSC_forest_model import TimeSeriesClassifierForest as tscf
from alg_repository.hydra_model import HydraCNNClassifier as hydrc
from alg_repository.all_estimators import AllEstimators as allest
from alg_repository.PMU_symmetrical_components import PMU_symmetrical_components as pmu_sc
from alg_repository.PMU_relative_unit import PMU_relative_units as pmu_ru

import json
import joblib
import numpy as np
import pandas as pd
import datetime
import os
import logging

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
                
        
        self.model_config = json.load(open('config//config.json'))

        self.exp_list = self.model_config['feature']

        self.fit_types = self.model_config['ml_model_type']

        self.exp_file_path = self.model_config['experiment_files_path']

        self.config = self.model_config['config_ml']

        self.aim_path = self.model_config['aim_methods_path']

        self.exp_name = self.model_config['exp_name']

        self.use_relative_units = self.model_config['use_relative_units']

        logging.basicConfig(filename=f'logging\\{self.exp_name}.log', filemode='w',
                             format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

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
            'hydra': hydrc,
            'all_est': allest
        }

        self.fit_type_config = {
            'forest': [
                'base_estimator',
                'n_estimators',
                'n_intervals',
                'min_interval_length',
                'max_interval_length',
                'time_limit_in_minutes',
                'contract_max_n_estimators',
                'random_state',
                'n_jobs',
                'parallel_backend'
            ],
            'hydra': [
                'n_kernels',
                'n_groups',
                'n_jobs',
                'random_state'
            ]
        }
        
        self = self._configuration()

        self = self._aimData()

        self = self._load_data()

        self = self.relative_unit_conversion()

        self = self._symmetrical_components_names()

        self = self._symmetrical_components()

        self = self._preprocessing()



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
                        logging.info(f"{attr_name} in {file_path}//{name} loaded successfully")
                        setattr(self, attr_name,
                                 rename_columns(di(',', file_path + '//' + name).main_process()))
                        print("Completed!",'\n')
                        logging.info("Completed!")

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
                    logging.info(f"{attr_name} in {file_path}//{name} loaded successfully")
                    setattr(self, attr_name,
                             rename_columns(di(',', file_path + '//' + name).main_process()))
                    print("Completed!",'\n')
                    logging.info("Completed!")
            
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
        logging.info(f"Loading data from {file_path}")
        print(file_path)
        file_names = os.listdir(file_path)
        logging.info(f"Files found: {file_names}")
        self = self._fileclassifier(file_path, file_names)

        return self
    
    def relative_unit_conversion(self):
        
        if self.use_relative_units == True:
            magnitude_list = []
            for exp_name in self.exp_list:
                if "seq" not in exp_name.split(sep="_") and "angle" not in exp_name.split(sep="_"):
                    print(exp_name)
                    magnitude_list.append(exp_name)

            for exp_name in magnitude_list:
                print(f"Converting {exp_name} to relative units")
                logging.info(f"Converting {exp_name} to relative units")
                setattr(self, exp_name, pmu_ru(
                    magnitude=getattr(self, exp_name),
                    measurement_type=exp_name.split("_")[0]).measurement())
                print("Completed!", "\n")
                logging.info("Completed!")
        else:
            print("Relative unit conversion is not needed", "\n")
            logging.info("Relative unit conversion is not needed")
            return self
                
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
            logging.info(f"Completed {name}")

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
        exp_names = ['exp ' + str(indx + 1) for indx in range(4608)]

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
        self.X = data
        
        return self
    
    def _aimData(self):
        """
        Loads and returns the best method from a file located in the 'method train result' directory.
        Returns:
            numpy.ndarray: The best method, transposed.
        """     
        file_names = os.listdir(self.aim_path)
        if len(file_names) == 0:
            raise FileNotFoundError('No files in the directory')
        else:
            for file in file_names:
                setattr(self, file.split('.')[0],
                    np.array(di(';',self.aim_path + file_names[0]).main_process()).transpose().flatten())
        return self

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
        for key in self.config.keys():
            if type(self.config[key]) == str:
                if self.config[key] == 'null':
                    self.config[key] = None
                elif self.config[key] == 'infinate':
                    self.config[key] = np.inf
        return self
    
    def TypeConfigParam(self, fit_type):
        config = {}
        if fit_type == 'forest':
            for key in self.fit_type_config[fit_type]:
                config[key] = self.config[key]
        elif fit_type == 'hydra':
            for key in self.fit_type_config[fit_type]:
                config[key] = self.config[key]
        return config
    
    def _timenow(self):
        """
        Returns the current date and time as a string in the format "dd_mm_YYYY_HH_MM".
        
        Returns:
            str: The current date and time as a string.
        """      
        return datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")

    def _save_model(self, model, exp_name, acc, report, timenow, fit_type):
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
                'model': self.fit_types,
                'feature': self.exp_list,
                'accuracy': acc,
                'report': report,
                'model address': adress + filename + '_' + fit_type + '.pkl'
            }

            if not os.path.exists(adress):
                os.makedirs(adress)

            with open(adress + filename + '_' + fit_type + '.json', 'w') as f:
                json.dump(model_description, f, indent=4)
                joblib.dump(model, model_description['model address'])
        return 

    def fit(self):
        """
        Trains a time series classification model and saves the trained model along with relevant metadata.
        
        Args:
            exp (str, optional): The name of the experiment or model being trained. If not provided, the `exp_type` attribute of the `LearnProcess` instance will be used.
            timenow (str, optional): The current date and time as a string in the format "dd_mm_YYYY_HH_MM". If not provided, it will be generated using the `_timenow()` method.
        
        Returns:
            tuple: A tuple containing the trained model, the accuracy score, and a report of the model's performance metrics.
        """
        timenow = self._timenow()
        X = self.X
        for aim_name in os.listdir(self.aim_path):
            for fit_type in self.fit_types:
                print('Training the model: ', fit_type)
                logging.info(f"Training model: {fit_type}")
                print('Aim: ', aim_name.split('.')[0])
                logging.info(f"Training model with aim: {aim_name.split('.')[0]}")
                Y = getattr(
                    self,
                    aim_name.split('.')[0]
                )
                model, acc, report = self.fit_library[
                    fit_type](self.TypeConfigParam(fit_type), X, Y, self.exp_name).main()
                self._save_model(model, self.exp_name, 
                                 acc, report, timenow, fit_type)
                
                logging.info(
                    f"Model training completed for {fit_type} with accuracy {acc}"
                )
                logging.info("Fin")
                del model, acc, report
        return "Fin"

