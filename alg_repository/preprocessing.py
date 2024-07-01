# Предварительная подготовка данных 
from alg_repository.data_importer import DataImporter as di
class dataPreprocessFromCSV():

    def __init__(self, path_name = None):
        self.path_name = path_name

    def data_import(self):
        if self.path_name is None:
            self.PMU_data = di().main_process()
        else:
            self.PMU_data = di(',', self.path_name).main_process()
        return self.PMU_data
    
    
    
  