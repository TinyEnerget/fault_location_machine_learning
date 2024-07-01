import tkinter as tk
from tkinter import filedialog
import pandas as pd

class DataImporter:

    def __init__(self,sep = ';', input_file_path = None):
        self.file_path = None
        self.file_type = None
        self.data = None
        self.sep = sep
        self.input_file_path = input_file_path
    

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
        if self.file_type == 'csv' or self.file_type == 'CSV':
            self.data = pd.read_csv(self.file_path, sep = self.sep)
        elif self.file_type == 'tsv':
            self.data = pd.read_csv(self.file_path, sep = self.sep)
        elif self.file_type == 'xlsx':
            self.data = pd.read_excel(self.file_path)
        elif self.file_type == 'xls':
            self.data = pd.read_excel(self.file_path)
        elif self.file_type == 'xlsm':
            self.data = pd.read_excel(self.file_path)
        elif self.file_type == 'xlsb':
            self.data = pd.read_excel(self.file_path)
        else:
            print('File type not supported')
        
        return self.data
    
    def main_process(self):
        self = DataImporter.select_file(self)
        return DataImporter.import_data(self)
