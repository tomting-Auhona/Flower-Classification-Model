import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
import cv2

# Load the trained model
model_path = "flower_v2.h5"
model = load_model(model_path)
labels = ['dandelion', 'daisy','tulip','sunflower','rose']
img_size = 224

# Define function to preprocess the image
def preprocess_image(image):
    img = cv2.resize(image, (img_size, img_size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define function to make predictions
def predict_flower(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    flower_class = np.argmax(prediction)
    return labels[flower_class]

# Define function to open file dialog and process image
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        predicted_flower = predict_flower(image)
        label_result.config(text=f"Predicted Flower Class: {predicted_flower}")

# Create Tkinter window
root = tk.Tk()
root.title("Flower Classification App")

# Set window size and position
window_width = 650
window_height = 700
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Set flower background
background_image = Image.open("flower_background.jpeg")
background_photo = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image=background_photo)
background_label.place(relwidth=1, relheight=1)

# Create header label
header_label = tk.Label(root, text="Flower Classification App", font=("Helvetica", 28, "bold"), bg="white", fg="#FF6347")
header_label.pack(pady=20)

# Create button to open file dialog
open_button = tk.Button(root, text="Open Image", command=open_image, font=("Helvetica", 16), bg="#4CAF50", fg="white", padx=20, pady=10)
open_button.pack(pady=20)

# Create label to display prediction result
label_result = tk.Label(root, text="", font=("Helvetica", 18, "italic"), bg="white", fg="#FF6347")
label_result.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
