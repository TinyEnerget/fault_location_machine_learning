import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import os
import sys

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Time series types fault location's methods classifier")
        self.root.geometry("800x600")
        
        # Set up theme
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create and set up widgets
        self.setup_widgets()
        
    def setup_widgets(self):
        # Header
        header_label = ctk.CTkLabel(self.main_frame, text="Time series classifier", font=("Helvetica", 24))
        header_label.pack(pady=20)
        instruction_lable = ctk.CTkLabel(self.main_frame, 
                                         text="Потом напаишу краткую инструкцию с параметрами", font=("Helvetica", 16))
        instruction_lable.pack(pady=20)
        
        # Buttons
        start_button = ctk.CTkButton(self.main_frame, text="Start", command=self.start_action)
        start_button.pack(pady=10)
        
        settings_button = ctk.CTkButton(self.main_frame, text="Settings", command=self.open_settings)
        settings_button.pack(pady=10)

        about_button = ctk.CTkButton(self.main_frame, text="About", command=self.open_about)
        about_button.pack(pady=10)
        
        exit_button = ctk.CTkButton(self.main_frame, text="Exit", command=self.exit_application)
        exit_button.pack(pady=10)
        
    def start_action(self):
        messagebox.showinfo("Action", "Starting the application...")
        
    def open_settings(self):
        settings_window = ctk.CTkToplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        
        settings_label = ctk.CTkLabel(settings_window, text="Settings", font=("Helvetica", 20))
        settings_label.pack(pady=20)

        #Text box
        settings_text = ctk.CTkTextbox(settings_window, height=10, width=40)
        settings_text.pack(pady=20)

        # Check box
        settings_check = ctk.CTkCheckBox(settings_window, text="Check me")
        settings_check.pack(pady=20)

        # Save button
        save_button = ctk.CTkButton(settings_window, text="Save", command=self.save_settings)
        save_button.pack(pady=20)
        
        # Add more settings widgets here
    
    def save_settings(self):
        messagebox.showinfo("Settings", "Settings saved!")

    def open_about(self):
        about_standart_window = ctk.CTkToplevel(self.root)
        about_standart_window.title("About")
        about_standart_window.geometry("400x300")

    def exit_application(self):
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            self.root.quit()

if __name__ == "__main__":
    root = ctk.CTk()
    app = GUI(root)
    root.mainloop()
