import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import os
import sys
import json

class GUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Time series types fault location's methods classifier")
        self.geometry(f"{1000}x{800}")

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Set up theme
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("dark-blue")
        ctk.deactivate_automatic_dpi_awareness()

        # create navigation frame
        self.navigation_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(5, weight=1)

        self.navigation_frame_label = ctk.CTkLabel(self.navigation_frame, text="Menu",
                                                             compound="left", font=ctk.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.start_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Start", 
                                          fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), 
                                          anchor="w", command=self.start_button_event)
        self.start_button.grid(row=1, column=0, sticky="ew")

        self.config_model_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Configure ML model",
                                                   fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                   anchor="w", command=self.config_model_button_event)
        self.config_model_button.grid(row=2, column=0, sticky="ew")

        self.train_ml_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Train ML model",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      anchor="w", command=self.train_ml_button_event)
        self.train_ml_button.grid(row=3, column=0, sticky="ew")

        self.frame_3_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Frame 3",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      anchor="w", command=self.frame_3_button_event)
        self.frame_3_button.grid(row=4, column=0, sticky="ew")

        self.appearance_mode_menu = ctk.CTkOptionMenu(self.navigation_frame, values=["System","Light", "Dark"],
                                                                command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=6, column=0, padx=20, pady=20, sticky="s")

        # create configure model frame
        self.config_model_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent", width=600, height=800)
        self.config_model_frame.grid_rowconfigure((5,10), weight=1)
        self.config_model_frame.grid_columnconfigure((6,7,9,10), weight=1)
        self.config_model_frame_label_intro = ctk.CTkLabel(self.config_model_frame, text="Configure your model", 
                                                         font=ctk.CTkFont(size=20, weight="bold"))
        self.config_model_frame_label_intro.grid(row=0, column=4, columnspan=2, padx=0, pady=20, sticky="nsew")

        self.config_model_frame_label_intro_1 = ctk.CTkLabel(self.config_model_frame, 
                                                            text="In this section you will be able to configure your model", 
                                                            compound= "center")
        self.config_model_frame_label_intro_1.grid(row=1, column=4,
                                                            columnspan=2)
        
        self.model_parameters_frame_switch_base_parameters = ctk.CTkSwitch(self.config_model_frame,
                                                                              text="Use base parameters",
                                                                              command=self.use_base_parameters_event)
        self.model_parameters_frame_switch_base_parameters.grid(row=2, column=4, padx=0, pady=20)

        self.model_parameters_frame_label = ctk.CTkLabel(self.config_model_frame, 
                                                              text="Model configuration", 
                                                              font=ctk.CTkFont(size=15, 
                                                                               weight='normal'))
        self.model_parameters_frame_label.grid(row=3, column=4, padx=0, pady=0, sticky="nsew")
        
        self.model_parameters_frame_label_1 = ctk.CTkLabel(self.config_model_frame, 
                                                              text="Feature configuration", 
                                                              font=ctk.CTkFont(size=15, 
                                                                               weight='normal'))
        self.model_parameters_frame_label_1.grid(row=3, column=5, padx=0, pady=0, sticky="nsew")

        # config model заполнение данными
        self.model_parameters_frame = ctk.CTkScrollableFrame(master = self.config_model_frame,
                                                   corner_radius=5, width=400, height=500,
                                                   fg_color="gray25")
        self.model_parameters_frame.grid(row=4, column=4, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.model_parameters_frame.grid_rowconfigure(20, weight=1)
        self.model_parameters_frame.grid_columnconfigure((0, 1), weight=1)
    
        model_parameters_lib = self.get_model_config()

        self.model_parameters_frame_fit_type_switches = []
        row_idx = 0
        for  key, value in model_parameters_lib["fit type"].items():
            row_idx += 1
            switch = ctk.CTkSwitch(self.model_parameters_frame, text=value, width=200)
            switch.grid(row=2 + row_idx, column=0, columnspan=2, padx=0, pady=10)
            self.model_parameters_frame_fit_type_switches.append(switch)
        

        self.model_parameters_frame_name_entry = ctk.CTkEntry(self.model_parameters_frame, 
                                                                    placeholder_text=model_parameters_lib["model name"], width=200)
        self.model_parameters_frame_name_entry.grid(row=5, column=0, columnspan=2, padx=0, pady=10)

        # forest parameters
        self.model_parameters_frame_fit_type_lables_forest = ctk.CTkLabel(self.model_parameters_frame,
                                                                          text="Forest parameters",
                                                                          font=ctk.CTkFont(size=15, weight='normal'))
        self.model_parameters_frame_fit_type_lables_forest.grid(row=6, column=0, columnspan=2, padx=0, pady=10)

        self.model_parameters_frame_base_estimator_lable = ctk.CTkLabel(self.model_parameters_frame, text="Base Estimator: ")
        self.model_parameters_frame_base_estimator_lable.grid(row=7, column=0, ipadx=0, pady=10)

        self.model_parameters_frame_base_estimator_option_menu = ctk.CTkOptionMenu(self.model_parameters_frame,
                                                                                   values=[
                                                                                       "null",
                                                                                       "BaseEstimator"
                                                                                   ], bg_color="transparent")
        self.model_parameters_frame_base_estimator_option_menu.grid(row=7, column=1, ipadx=20, pady=10)

        self.model_parameters_frame_forest_param_entries = []
        row_idx = 0
        for key, value in model_parameters_lib["model parameters"]["forest"]["entery"].items():
            entry = ctk.CTkEntry(self.model_parameters_frame, 
                                 placeholder_text=key+": "+str(model_parameters_lib["base parameters"][key]), width=200)
            entry.grid(row=8 + row_idx, column=0, columnspan=2, padx=0, pady=10)
            self.model_parameters_frame_forest_param_entries.append(entry)
            row_idx += 1
        
        last_row = row_idx + 8

        # hydra parameters entery
        self.model_parameters_frame_fit_type_lables_hydra = ctk.CTkLabel(self.model_parameters_frame,
                                                                          text="Hydra parameters",
                                                                          font=ctk.CTkFont(size=15, weight='normal'))
        self.model_parameters_frame_fit_type_lables_hydra.grid(row=last_row+1, column=0, columnspan=2, padx=0, pady=10)
        self.model_parameters_frame_hydra_param_entries = []
        row_idx = 0
        for key, value in model_parameters_lib["model parameters"]["hydra"].items():
            entry = ctk.CTkEntry(self.model_parameters_frame, 
                                 placeholder_text=key+": "+str(model_parameters_lib["base parameters"][key]), width=200)
            entry.grid(row=last_row+2 + row_idx, column=0, columnspan=2, padx=0, pady=10)
            self.model_parameters_frame_hydra_param_entries.append(entry)
            row_idx += 1

        last_row = row_idx + last_row

        self.config_model_frame_save_config_button = ctk.CTkButton(self.config_model_frame,
                                                                  text="Save config",
                                                                  command=self.save_config_event)
        self.config_model_frame_save_config_button.grid(row=5, column=4, padx=0, pady=10)


        self.features_config_frame = ctk.CTkScrollableFrame(master = self.config_model_frame,
                                                            corner_radius=5, width=320, height=500,
                                                            fg_color="gray25")
        self.features_config_frame.grid(row=4, column=5, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.features_config_frame.grid_rowconfigure(20, weight=1)
        self.features_config_frame.grid_columnconfigure(0, weight=1)
        self.features_list = self.get_features_config()
        self.features_config_frame_features_chk_boxes = []
        row_idx = 0
        for feature in self.features_list:
            chk_box = ctk.CTkCheckBox(self.features_config_frame, text=feature, width=300 )
            chk_box.grid(row=row_idx, column=0, padx=10, pady=10)
            chk_box.select()
            self.features_config_frame_features_chk_boxes.append(chk_box)
            row_idx += 1


        # create train ml model frame
        self.train_ml_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent", width=600, height=800)
        self.train_ml_frame.grid_rowconfigure(20, weight=1)
        self.train_ml_frame.grid_columnconfigure(2, weight=1)

        self.train_ml_frame_label_intro = ctk.CTkLabel(self.train_ml_frame, text="Train your model", 
                                                         font=ctk.CTkFont(size=20, weight="bold"))
        self.train_ml_frame_label_intro.grid(row=0, column=0, columnspan=3, padx=0, pady=20, sticky="nsew")

        self.train_ml_frame_label_intro_1 = ctk.CTkLabel(self.train_ml_frame, 
                                                            text="In this section you will be able to train your model", 
                                                            compound= "center")
        self.train_ml_frame_label_intro_1.grid(row=1, column=0, columnspan=3)

        self.train_ml_frame_textBox = ctk.CTkTextbox(self.train_ml_frame, height=400, width=400)
        self.train_ml_frame_textBox.grid(row=2, column=0, columnspan=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.load_json_to_treeview()


        # create third frame
        self.third_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        
        # create start frame
        self.start_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.start_frame.grid_columnconfigure(0, weight=1)
        self.start_frame_label = ctk.CTkLabel(self.start_frame, text="Start", font=ctk.CTkFont(size=20, weight="bold"))
        self.start_frame_label.grid(row=0, column=0, padx=20, pady=10)
        self.start_frame_label = ctk.CTkLabel(self.start_frame, text="There will be little instruction and intruduction", font=ctk.CTkFont(size=15, weight='normal'))
        self.start_frame_label.grid(row=1, column=0, padx=20, pady=10)

        self.select_frame_by_name("start")


    def load_json_to_treeview(self):
        with open("config\\config.json", "r") as file:
            data = json.load(file)

        style = ttk.Style()
        style.configure("Treeview", 
                background="gray75",
                foreground="gray25",
                fieldbackground="gray75",
                rowheight=25,
                font=ctk.CTkFont(size=15, weight='normal'))
                
        style.map('Treeview', background=[('selected', 'gray25')])
        tree = ttk.Treeview(self.train_ml_frame, style="Treeview")
        tree["columns"] = ("Value")
        tree.column("#0", width=400, minwidth=400)
        tree.column("Value", width=200, minwidth=200)
        tree.heading("#0", text="Key")
        tree.heading("Value", text="Value")

        def insert_into_treeview(parent, key, value):
            if isinstance(value, dict):
                node = tree.insert(parent, "end", text=key, open=True)
                for k, v in value.items():
                    insert_into_treeview(node, k, v)
            elif isinstance(value, list):
                node = tree.insert(parent, "end", text=key, open=True)
                for i, item in enumerate(value):
                    insert_into_treeview(node, "", item)
            else:
                tree.insert(parent, "end", text=key, values=(value,))

        for key, value in data.items():
            insert_into_treeview("", key, value)

        tree.grid(row=2, column=0, columnspan=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        


    def get_model_config(self) -> dict:
        """
        TO DO:
        - n_intervals: write additional parameters for this parameter in subscriptions of this parameters-
        """
        self.ml_config = json.load(open("config base\\config_base.json"))
        self.base_config = self.ml_config["config_ml"]
        model_parameters_lib = {
            "fit type":{
                "forest": "Time Series Forest Classifier",
                "hydra": "Hydra Classifier"
                },
            "model name": "Model name",
            "base parameters": self.base_config,
            "model parameters": {
                "forest": {
                    "option menu": {
                        "base_estimator": "The base estimator from which the ensemble is built.",    
                    },
                    "entery": {
                        "n_estimators": "Number of trees",
                        "n_intervals": "Number of intervals",
                        "min_interval_length": "Minimum interval length",
                        "max_interval_length": "Maximum interval length",
                        "time_limit_in_minutes": "Time limit in minutes",
                        "contract_max_n_estimators": "Maximum number of estimators in the ensemble",
                        "n_jobs": "Number of jobs to run in parallel for both `fit` and `predict`.",
                        "random_state": "Seed for random number generation",
                        "parallel_backend": "Which backend to use to train models in parallel."
                    }
                },
                "hydra": {
                    "n_kernels": "Number of kernels",
                    "n_groups": "Number of groups",
                    "random_state": "Seed for random number generation",
                    "n_jobs": "Number of jobs to run in parallel for both `fit` and `predict`."
                }
            }
        }
        return model_parameters_lib
    
    def get_features_config(self) -> dict:
        features_config = self.ml_config["feature"]
        return features_config

    def use_base_parameters_event(self):
        print(self.model_parameters_frame_switch_base_parameters.get())
        if self.model_parameters_frame_switch_base_parameters.get() == 1:
            self.model_parameters_frame.configure(fg_color="gray75")
            self.model_parameters_frame_fit_type_switches[0].configure(fg_color="gray75", state="disabled")
            self.model_parameters_frame_fit_type_switches[1].configure(fg_color="gray75", state="disabled")
            self.model_parameters_frame_base_estimator_option_menu.configure(fg_color="gray75", state="disabled")
            for idx in range(len(self.model_parameters_frame_forest_param_entries)):
                self.model_parameters_frame_forest_param_entries[idx].configure(fg_color="gray75", state="disabled")
            for idx in range(len(self.model_parameters_frame_hydra_param_entries)):
                self.model_parameters_frame_hydra_param_entries[idx].configure(fg_color="gray75", state="disabled")
            
        elif self.model_parameters_frame_switch_base_parameters.get() == 0:
            self.model_parameters_frame.configure(fg_color="gray25")
            self.model_parameters_frame_fit_type_switches[0].configure(fg_color="gray25", state="normal")
            self.model_parameters_frame_fit_type_switches[1].configure(fg_color="gray25", state="normal")
            self.model_parameters_frame_base_estimator_option_menu.configure(fg_color="gray25", state="normal")
            for idx in range(len(self.model_parameters_frame_forest_param_entries)):
                self.model_parameters_frame_forest_param_entries[idx].configure(fg_color="gray25", state="normal")
            for idx in range(len(self.model_parameters_frame_hydra_param_entries)):
                self.model_parameters_frame_hydra_param_entries[idx].configure(fg_color="gray25", state="normal")
    
    def validate_input(self, value, expected_type):
        try:
            if expected_type == int:
                return int(value)
            elif expected_type == float:
                return float(value)
            elif expected_type == str:
                return str(value)
            else:
                return False
        except ValueError:
            #messagebox.showerror("Title", "This is an error message")
            return "null data"
    def save_config_event(self):
        print("save config")
        try:
            model_name = self.model_parameters_frame_name_entry.get()

            if self.model_parameters_frame_fit_type_switches[0].get() == 1:
                fit_type = ["forest"]
                config_ml = {
                    "base_estimator": self.validate_input(self.model_parameters_frame_forest_param_entries[0].get(), str),
                    "n_estimators": self.validate_input(self.model_parameters_frame_forest_param_entries[1].get(), int),
                    "n_intervals": self.validate_input(self.model_parameters_frame_forest_param_entries[2].get(), str),
                    "min_interval_length": self.validate_input(self.model_parameters_frame_forest_param_entries[3].get(), int),
                    "max_interval_length": self.validate_input(self.model_parameters_frame_forest_param_entries[4].get(), int),
                    "time_limit_in_minutes": self.validate_input(self.model_parameters_frame_forest_param_entries[5].get(), str),
                    "contract_max_n_estimators": self.validate_input(self.model_parameters_frame_forest_param_entries[6].get(), int),
                    "random_state": self.validate_input(self.model_parameters_frame_forest_param_entries[7].get(), str),
                    "n_jobs": self.validate_input(self.model_parameters_frame_forest_param_entries[8].get(), int),
                    "parallel_backend": self.validate_input(self.model_parameters_frame_forest_param_entries[9].get(), str)
                    }
            elif self.model_parameters_frame_fit_type_switches[1].get() == 1:
                fit_type = ["hydra"]
                config_ml = {
                    "n_kernels": self.validate_input(self.model_parameters_frame_hydra_param_entries[0].get(), str),
                    "n_groups": self.validate_input(self.model_parameters_frame_hydra_param_entries[1].get(), str),
                    "random_state": self.validate_input(self.model_parameters_frame_hydra_param_entries[2].get(), str),
                    "n_jobs": self.validate_input(self.model_parameters_frame_hydra_param_entries[3].get(), str)
                    }
            elif self.model_parameters_frame_fit_type_switches[0].get() == 1 and self.model_parameters_frame_fit_type_switches[1].get() == 1:
                fit_type = ["forest", "hydra"]
                config_ml = {
                        "base_estimator": self.model_parameters_frame_forest_param_entries[0].get(),
                        "n_estimators": self.model_parameters_frame_forest_param_entries[1].get(),
                        "n_intervals": self.model_parameters_frame_forest_param_entries[2].get(),
                        "min_interval_length": self.model_parameters_frame_forest_param_entries[3].get(),
                        "max_interval_length": self.model_parameters_frame_forest_param_entries[4].get(),
                        "time_limit_in_minutes": self.model_parameters_frame_forest_param_entries[5].get(),
                        "contract_max_n_estimators": self.model_parameters_frame_forest_param_entries[6].get(),
                        "random_state": self.model_parameters_frame_forest_param_entries[7].get(),
                        "n_jobs": self.model_parameters_frame_forest_param_entries[8].get(),
                        "parallel_backend": self.model_parameters_frame_forest_param_entries[9].get(),
                        "n_kernels": self.model_parameters_frame_hydra_param_entries[0].get(),
                        "n_groups": self.model_parameters_frame_hydra_param_entries[1].get(),
                    }
            elif self.model_parameters_frame_switch_base_parameters.get() == 1:
                fit_type = self.ml_config["ml_model_type"]
                config_ml = self.base_config

            feature = []
            for idx in range(len(self.features_config_frame_features_chk_boxes)):
                if self.features_config_frame_features_chk_boxes[idx].get() == 1:
                    feat = self.features_list[idx]
                    feature.append(feat)

            config = {
                "exp_name": model_name,

                "ml_model_type": fit_type,

                "feature": feature,

                "config_ml": config_ml,

                "experiment_files_path": "CSV PMU 100fps",

                "aim_methods_path": "aim methods\\"
            }
            adress = "C:\\Users\\Vlad Titov\\Desktop\\Work\\fault_location_machine_learning\\config\\"
            filename= "config"        
            with open(adress + filename + '.json', 'w') as f:
                json.dump(config, f, indent=4)
        except:
            Warning("Error in saving config")


    def select_frame_by_name(self, name):
        # set button color for selected button
        self.start_button.configure(fg_color=("gray75", "gray25") if name == "start" else "transparent")
        self.config_model_button.configure(fg_color=("gray75", "gray25") if name == "config_model" else "transparent")
        self.train_ml_button.configure(fg_color=("gray75", "gray25") if name == "train_ml" else "transparent")
        self.frame_3_button.configure(fg_color=("gray75", "gray25") if name == "frame_3" else "transparent")

        # show selected frame
        if name == "start":
            self.start_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.start_frame.grid_forget()
        if name == "config_model":
            self.config_model_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.config_model_frame.grid_forget()
        if name == "train_ml":
            self.train_ml_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.train_ml_frame.grid_forget()
        if name == "frame_3":
            self.third_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.third_frame.grid_forget()
    
    def start_button_event(self):
        self.select_frame_by_name("start")
        
    def config_model_button_event(self):
        self.select_frame_by_name("config_model")

    def train_ml_button_event(self):
        self.select_frame_by_name("train_ml")

    def frame_3_button_event(self):
        self.select_frame_by_name("frame_3")

    def change_appearance_mode_event(self, new_appearance_mode):
        ctk.set_appearance_mode(new_appearance_mode)
    
if __name__ == "__main__":
    app = GUI()
    app.mainloop()
