import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import os
import sys

from cycler import V

class GUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Time series types fault location's methods classifier")
        self.geometry("800x600")

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Set up theme
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("dark-blue")
        ctk.deactivate_automatic_dpi_awareness()

        # create navigation frame
        self.navigation_frame = ctk.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(5, weight=1)

        self.navigation_frame_label = ctk.CTkLabel(self.navigation_frame, text="Menu",
                                                             compound="left", font=ctk.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.start_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Start", 
                                          fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), 
                                          anchor="w", command=self.start_button_event)
        self.start_button.grid(row=1, column=0, sticky="ew")

        self.train_model_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Train ML model",
                                                   fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                   anchor="w", command=self.train_model_button_event)
        self.train_model_button.grid(row=2, column=0, sticky="ew")

        self.frame_2_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Frame 2",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      anchor="w", command=self.frame_2_button_event)
        self.frame_2_button.grid(row=3, column=0, sticky="ew")

        self.frame_3_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Frame 3",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      anchor="w", command=self.frame_3_button_event)
        self.frame_3_button.grid(row=4, column=0, sticky="ew")

        self.appearance_mode_menu = ctk.CTkOptionMenu(self.navigation_frame, values=["Light", "Dark", "System"],
                                                                command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=6, column=0, padx=20, pady=20, sticky="s")

        
        
        # create train model frame
        self.train_model_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        #self.home_frame.grid_columnconfigure(0, weight=1)

        # create second frame
        self.second_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")

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

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.start_button.configure(fg_color=("gray75", "gray25") if name == "start" else "transparent")
        self.train_model_button.configure(fg_color=("gray75", "gray25") if name == "train_model" else "transparent")
        self.frame_2_button.configure(fg_color=("gray75", "gray25") if name == "frame_2" else "transparent")
        self.frame_3_button.configure(fg_color=("gray75", "gray25") if name == "frame_3" else "transparent")

        # show selected frame
        if name == "start":
            self.start_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.start_frame.grid_forget()
        if name == "train_model":
            self.train_model_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.train_model_frame.grid_forget()
        if name == "frame_2":
            self.second_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.second_frame.grid_forget()
        if name == "frame_3":
            self.third_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.third_frame.grid_forget()
    
    def start_button_event(self):
        self.select_frame_by_name("start")
        self.start_frame_label = ctk.CTkLabel(self.start_frame, text="Start", font=ctk.CTkFont(size=20, weight="bold"))
        self.start_frame_label.grid(row=0, column=0, padx=20, pady=10)
        self.start_frame_label = ctk.CTkLabel(self.start_frame, text="There will be little instruction and intruduction", font=ctk.CTkFont(size=15, weight='normal'))
        self.start_frame_label.grid(row=1, column=0, padx=20, pady=10)

    def train_model_button_event(self):
        self.select_frame_by_name("train_model")
        self.train_model_frame.grid_columnconfigure([0,0], weight=1)
        self.train_model_frame_label_intro = ctk.CTkLabel(self.train_model_frame, text="Configure your model", 
                                                         font=ctk.CTkFont(size=20, weight="bold"), compound= "left")
        self.train_model_frame_label_intro.grid(row=0, padx=20, pady=10)

        self.train_model_frame_label_intro_1 = ctk.CTkLabel(self.train_model_frame, 
                                                            text="In this section you will be able to configure your model", compound= "left")
        self.train_model_frame_label_intro_1.grid(row=1, padx=20, pady=0)

        # config model заполнение данными
        self.train_model_frame_label_name = ctk.CTkLabel(self.train_model_frame, text="Model name", compound= "left")
        self.train_model_frame_label_name.grid(row=2, column=0, padx=20, pady=10)
        self.train_model_frame_label_name_entry = ctk.CTkEntry(self.train_model_frame, placeholder_text="Enter model name")
        self.train_model_frame_label_name_entry.grid(row=2, column=1, padx=0, pady=10)
        #self.home_frame_button_1 = ctk.CTkButton(self.home_frame, text="Penis")
        #self.home_frame_button_1.grid(row=1, column=0, padx=20, pady=10)
        #self.home_frame_button_2 = ctk.CTkButton(self.home_frame, text="CTkButton", compound="right")
        #self.home_frame_button_2.grid(row=2, column=0, padx=20, pady=10)
        #self.home_frame_button_3 = ctk.CTkButton(self.home_frame, text="CTkButton", compound="top")
        #self.home_frame_button_3.grid(row=3, column=0, padx=20, pady=10)
        #self.home_frame_button_4 = ctk.CTkButton(self.home_frame, text="CTkButton", compound="bottom", anchor="w")
        #self.home_frame_button_4.grid(row=4, column=0, padx=20, pady=10)

    def frame_2_button_event(self):
        self.select_frame_by_name("frame_2")

    def frame_3_button_event(self):
        self.select_frame_by_name("frame_3")

    def change_appearance_mode_event(self, new_appearance_mode):
        ctk.set_appearance_mode(new_appearance_mode)
    
if __name__ == "__main__":
    app = GUI()
    app.mainloop()
