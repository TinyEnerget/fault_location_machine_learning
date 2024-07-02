import tkinter as tk
from tkinter import filedialog
import pandas as pd

class DataImporter:

    def __init__(self,sep = ';', input_file_path = None):
        self.file_path = str
        self.file_type = str
        self.data = pd.DataFrame()
        self.sep = sep
        self.input_file_path = input_file_path
        self.file_type_mapping = {
            'csv': pd.read_csv,
            'tsv': pd.read_csv,
            'xlsx': pd.read_excel,
            'xls': pd.read_excel,
            'xlsm': pd.read_excel,
            'xlsb': pd.read_excel
            }
    

    def select_file(self):

        if self.input_file_path is None:
            filetypes = [('Exel files', ['*.xlsx', '*.xls', '*.xlsm', '*.xlsb']),
                          ('CSV/TSV files', ['*.csv', '*.tsv']), 
                          ('All files', '*.*')]
            root = tk.Tk()
            root.withdraw()
            self.file_path = filedialog.askopenfilename(
                title='Select a file',
                filetypes=filetypes
            )
            root.destroy()
        else:
            self.file_path = self.input_file_path
        
        self.file_type = self.file_path.split('.')[-1]
        return self
    
        

    def import_data(self) -> pd.DataFrame:
        file_path = self.file_path
        file_type_mapping = self.file_type_mapping
        file_type = str(self.file_type)
        if file_type.lower() in file_type_mapping:
            self.data = file_type_mapping[file_type](file_path, sep=self.sep)
        else:
            print('File type not supported')
        return self.data

    #def import_data(self) -> pd.DataFrame:
    #    file_path = self.file_path
    #    if self.file_type == 'csv' or self.file_type == 'CSV':
    #        self.data = pd.read_csv(file_path, sep = self.sep) # type: ignore
    #    elif self.file_type == 'tsv':
    #        self.data = pd.read_csv(file_path, sep = self.sep)
    #    elif self.file_type == 'xlsx':
    #        self.data = pd.read_excel(file_path)
    #    elif self.file_type == 'xls':
    #        self.data = pd.read_excel(file_path)
    #    elif self.file_type == 'xlsm':
    #        self.data = pd.read_excel(file_path)
    #    elif self.file_type == 'xlsb':
    #        self.data = pd.read_excel(file_path)
    #    else:
    #        print('File type not supported')
    #    
    #    return self.data
    
    def main_process(self):
        self = DataImporter.select_file(self)
        return DataImporter.import_data(self)
