import pandas as pd
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shutil

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((100, 100))  # Resize for consistency
    return np.array(img).flatten()  # Flatten to 1D array

def train_light_dark_classifier(train_folder):
    images = []
    labels = []
    for label in ['light', 'dark']:
        folder = os.path.join(train_folder, label)
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} does not exist.")
            continue
        try:
            for filename in os.listdir(folder):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder, filename)
                    img_array = load_and_preprocess_image(img_path)
                    images.append(img_array)
                    labels.append(label)
        except Exception as e:
            print(f"Error processing folder {folder}: {str(e)}")
    
    if not images:
        raise ValueError("No images found in the training folder.")
    
    X = np.array(images)
    y = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    return model

def analyze_new_images(model, input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            img_array = load_and_preprocess_image(img_path)
            prediction = model.predict([img_array])[0]
            
            # Rename the file
            new_filename = f"{prediction}_{filename}"
            new_path = os.path.join(input_folder, new_filename)
            os.rename(img_path, new_path)
            
            # Move the file to the output folder
            dest_path = os.path.join(output_folder, new_filename)
            shutil.move(new_path, dest_path)
            
            print(f"Processed {filename}: Classified as {prediction}")

# Usage
train_folder = 'Examples/Light Classifier/train_folder'  # Contains 'light' and 'dark' subfolders with labeled images
input_folder = 'Examples/Light Classifier/input_folder'  # Where new images will be placed for analysis
output_folder = 'Examples/Light Classifier/output_folder'  # Where processed images will be moved

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Train the model
try:
    print(f"Training folder path: {os.path.abspath(train_folder)}")
    model = train_light_dark_classifier(train_folder)
except Exception as e:
    print(f"Error training the model: {str(e)}")
    print("Please check if the train_folder path is correct and contains 'light' and 'dark' subfolders with images.")
    exit(1)

# Set up a loop to continuously check for new images
import time

while True:
    analyze_new_images(model, input_folder, output_folder)
    time.sleep(60)  # Check every 60 seconds
    # Break out of the infinite loop
    break

# Import necessary libraries for GUI
import tkinter as tk
from tkinter import filedialog, messagebox

def choose_folder(title):
    return filedialog.askdirectory(title=title)

def start_analysis():
    global train_folder, input_folder, output_folder, model
    
    # Disable the start button to prevent multiple clicks
    start_button.config(state=tk.DISABLED)
    
    # Get the string values from the StringVar objects
    train_folder_path = train_folder.get()
    input_folder_path = input_folder.get()
    output_folder_path = output_folder.get()
    
    # Train the model
    try:
        print(f"Training folder path: {os.path.abspath(train_folder_path)}")
        model = train_light_dark_classifier(train_folder_path)
    except Exception as e:
        messagebox.showerror("Error", f"Error training the model: {str(e)}\n\nPlease check if the train_folder path is correct and contains 'light' and 'dark' subfolders with images.")
        start_button.config(state=tk.NORMAL)
        return
    
    # Start the analysis
    try:
        analyze_new_images(model, input_folder_path, output_folder_path)
        messagebox.showinfo("Analysis Complete", "Image analysis has been completed.")
    except Exception as e:
        messagebox.showerror("Error", f"Error during analysis: {str(e)}")
    
    # Re-enable the start button
    start_button.config(state=tk.NORMAL)

# Create the main window
root = tk.Tk()
root.title("Light/Dark Image Classifier")
root.geometry("400x300")

# Create and pack widgets
tk.Label(root, text="Light/Dark Image Classifier", font=("Arial", 16)).pack(pady=10)

train_folder = tk.StringVar()
input_folder = tk.StringVar()
output_folder = tk.StringVar()

tk.Button(root, text="Choose Train Folder", command=lambda: train_folder.set(choose_folder("Select Train Folder"))).pack(pady=5)
tk.Button(root, text="Choose Input Folder", command=lambda: input_folder.set(choose_folder("Select Input Folder"))).pack(pady=5)
tk.Button(root, text="Choose Output Folder", command=lambda: output_folder.set(choose_folder("Select Output Folder"))).pack(pady=5)

start_button = tk.Button(root, text="Start Analysis", command=start_analysis)
start_button.pack(pady=20)

# Start the GUI event loop
root.mainloop()
