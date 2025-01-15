import torch
import numpy as np
import matplotlib.pyplot as plt
from auto_vae import VAE, generate_reconstructions, normalize_new_tensor
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Global variable to store loaded data
hsi_tensor = None  
scatter_points = []  
model = None  
current_sample_index = 0 
good_count = 0  
bad_count = 0  

# load the data
def load_data():
    global hsi_tensor
    # Load normalization components
    with open('normalization_components.pkl', 'rb') as f:
        components = pickle.load(f)
    components = {key: torch.tensor(value) for key, value in components.items()}

    # Load the input tensor and normalize
    data_dir = filedialog.askopenfilename(title="Select HSI Tensor File", filetypes=[("PT files", "*.pt")])
    if data_dir:
        hsi_tensor = torch.load(data_dir)
        hsi_tensor = normalize_new_tensor(hsi_tensor, components)
        hsi_tensor = hsi_tensor.permute(0, 3, 1, 2)  

# process the data
def process_data():
    global hsi_tensor, model, current_sample_index, good_count, bad_count
    if hsi_tensor is None:
        messagebox.showwarning("Load Data", "Please load HSI Tensor data first.")
        return  

    # Define the model and load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_channels=4).to(device)
    model.load_state_dict(torch.load('auto_vae2.pth'))
    model.eval()

    # Prepare data for multiple samples and run the model
    hsi_tensor = hsi_tensor.to(device)
    current_sample_index = 0
    good_count = 0
    bad_count = 0
    process_sample()

# process a single sample
def process_sample():
    global current_sample_index, scatter_points, model, good_count, bad_count

    if model is None or hsi_tensor is None:
        messagebox.showwarning("Load Data", "Please load HSI Tensor data and model first.")
        return

    if current_sample_index >= len(hsi_tensor):
        messagebox.showinfo("Completed", "All samples have been processed.")
        return

    
    sample = hsi_tensor[current_sample_index].unsqueeze(0)
    original, reconstructions, _, _, _ = generate_reconstructions(model, sample, 1)

    
    error = (original - reconstructions).abs().squeeze().cpu()

    
    reconstruction_loss = error.sum().item()  

    
    if reconstruction_loss > 1900:
        anomaly_label.config(text=f"Sample {current_sample_index+1}: Anomalous Tomato", fg="red")
        bad_count += 1
    else:
        anomaly_label.config(text=f"Sample {current_sample_index+1}: Good Tomato", fg="green")
        good_count += 1

    # Update the good and bad counts
    good_bad_count_label.config(text=f"Good Tomatoes: {good_count} | Anomalous Tomatoes: {bad_count}")

    # Visualize the error overlayed on the original image
    original_img = original.squeeze().cpu().permute(1, 2, 0)[:, :, :3].numpy()  # Convert to NumPy array
    error_img = error.permute(1, 2, 0).numpy()

    # Define a threshold for highlighting errors
    threshold = 0.02  # Adjust this value based on your specific use case
    error_highlight = (error_img > threshold).astype(np.float32)

    # Create a red overlay where the error exceeds the threshold
    red_overlay = np.zeros_like(original_img)
    red_overlay[:, :, 0] = error_highlight[:, :, 0]  # Red channel

    # Combine original image with red overlay
    highlighted_img = np.clip(original_img + red_overlay, 0, 1)  # Ensure values stay within [0, 1]

    # Show images in the GUI
    display_images(original_img, highlighted_img)

    # Update scatter points and plot
    scatter_points.append(reconstruction_loss)
    display_scatter_plot()

    # Move to the next sample when "Next" is clicked
    current_sample_index += 1

# Function to display images in the canvas
def display_images(original_img, highlighted_img):
    # Convert numpy arrays to images
    original_img = (original_img * 255).astype(np.uint8)
    highlighted_img = (highlighted_img * 255).astype(np.uint8)

    original_image = Image.fromarray(original_img)
    highlighted_image = Image.fromarray(highlighted_img)

    # Resize images to fit the canvas
    original_image = original_image.resize((300, 300))
    highlighted_image = highlighted_image.resize((300, 300))

    # Create PhotoImage objects
    original_photo = ImageTk.PhotoImage(original_image)
    highlighted_photo = ImageTk.PhotoImage(highlighted_image)

    # Clear existing image on the canvas and display new images
    original_canvas.delete("all")
    original_canvas.create_image(0, 0, anchor=tk.NW, image=original_photo)
    
    highlighted_canvas.delete("all")
    highlighted_canvas.create_image(0, 0, anchor=tk.NW, image=highlighted_photo)

    # Keep a reference to avoid garbage collection
    original_canvas.image = original_photo
    highlighted_canvas.image = highlighted_photo

# Function to display the scatter plot
def display_scatter_plot():
    # Clear the scatter canvas before drawing a new plot
    scatter_canvas.delete("all")
    
    # Create a scatter plot using matplotlib
    plt.figure(figsize=(3, 3))
    plt.scatter(range(len(scatter_points)), scatter_points, c=['green' if x < 1900 else 'red' for x in scatter_points], s=100)
    plt.axhline(y=1900, color='blue', linestyle='--', label='Threshold')
    plt.ylim(500, 3500)
    plt.xlabel('Sample')
    plt.ylabel('Reconstruction Loss')
    plt.title('Reconstruction Loss vs. Threshold')
    plt.legend(loc='upper right')
    
    # Save the plot to a temporary image file
    plt.tight_layout()
    plt.savefig('scatter_plot.png', bbox_inches='tight', transparent=True)
    plt.close()

    # Load the saved scatter plot image
    scatter_image = Image.open('scatter_plot.png')
    scatter_image = scatter_image.resize((300, 300))
    scatter_photo = ImageTk.PhotoImage(scatter_image)
    
    # Display the scatter plot on the canvas
    scatter_canvas.create_image(0, 0, anchor=tk.NW, image=scatter_photo)
    scatter_canvas.image = scatter_photo

# Create the GUI
root = tk.Tk()
root.title("HSI Anomaly Detector")
root.configure(bg="#F5F5F5")  # Set background color for the window

# Add a frame for better layout
frame = tk.Frame(root, padx=20, pady=20, bg="#F5F5F5")
frame.pack()

# Add a heading for the GUI
heading_label = tk.Label(frame, text="Tomato Anomaly Detector (QualiCrop Demo)", font=("Helvetica", 20, "bold"), bg="#F5F5F5", fg="blue")
heading_label.grid(row=0, columnspan=2, pady=10)

# Create a button to load the data
load_button = tk.Button(frame, text="Load HSI Tensor", command=load_data, width=20, bg="#4CAF50", fg="white", font=("Helvetica", 12))
load_button.grid(row=1, column=0, pady=10)

# Create a button to process the data
process_button = tk.Button(frame, text="Process HSI Tensor", command=process_data, width=20, bg="#2196F3", fg="white", font=("Helvetica", 12))
process_button.grid(row=1, column=1, pady=10)

# Create a button for the "Next" functionality
next_button = tk.Button(frame, text="Next", command=process_sample, width=20, bg="#FF9800", fg="white", font=("Helvetica", 12))
next_button.grid(row=2, columnspan=2, pady=10)

# Create a label for anomaly indication
anomaly_label = tk.Label(frame, text="", font=("Helvetica", 16), bg="#F5F5F5")
anomaly_label.grid(row=3, columnspan=2, pady=10)

# Create a label to show counts of good and bad samples
good_bad_count_label = tk.Label(frame, text="Good Tomatoes: 0 | Anomalous Tomatoes: 0", font=("Helvetica", 14), bg="#F5F5F5")
good_bad_count_label.grid(row=4, columnspan=2, pady=5)

# Create canvas elements for displaying images
original_canvas = tk.Canvas(frame, width=300, height=300, bg="#E0E0E0")
original_canvas.grid(row=5, column=0, padx=10, pady=10)
highlighted_canvas = tk.Canvas(frame, width=300, height=300, bg="#E0E0E0")
highlighted_canvas.grid(row=5, column=1, padx=10, pady=10)

# Create a canvas for displaying the scatter plot
scatter_canvas = tk.Canvas(frame, width=300, height=300, bg="#E0E0E0")
scatter_canvas.grid(row=6, columnspan=2, padx=10, pady=10)

# Start the main loop
root.mainloop()
