import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class HSIAnalysisApp:
    def __init__(self, root, hsi_data, rgb_image):
        self.root = root
        self.root.title("Hyperspectral Image Analysis")
        
        # Data
        self.hsi_data = hsi_data  # Shape should be (height, width, bands)
        self.rgb_image = rgb_image
        
        # Selected points
        self.normal_point = None
        self.anomalous_point = None
        self.spectrum_history = {'normal': [], 'anomalous': []}
        
        # Animation control
        self.animation_interval = 20  # ms between frames
        self.animation_frames = 15  # frames for transition
        
        # Create GUI layout with consistent sizing
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights for proper expansion
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # RGB Image Display (left column)
        rgb_frame = ttk.LabelFrame(main_frame, text="RGB Image")
        rgb_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        self.fig_rgb = plt.Figure(figsize=(5, 5), dpi=100)
        self.ax_rgb = self.fig_rgb.add_subplot(111)
        self.ax_rgb.imshow(self.rgb_image)
        self.ax_rgb.set_title("Click normal (green) then anomalous (red) pixels")
        self.ax_rgb.axis('off')
        
        self.canvas_rgb = FigureCanvasTkAgg(self.fig_rgb, master=rgb_frame)
        self.canvas_rgb.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_rgb.mpl_connect('button_press_event', self.on_rgb_click)
        
        # Right column container for spectra
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)
        
        # Normal Spectrum (top right)
        normal_frame = ttk.LabelFrame(right_frame, text="Normal Spectrum")
        normal_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        self.fig_normal = plt.Figure(figsize=(5, 2.5), dpi=100)  # Same size as anomalous
        self.ax_normal = self.fig_normal.add_subplot(111)
        self.ax_normal.set_xlabel("Wavelength Band")
        self.ax_normal.set_ylabel("Reflectance")
        self.line_normal, = self.ax_normal.plot([], [], 'g-', alpha=0.7)
        self.ax_normal.set_xlim(0, self.hsi_data.shape[2])
        self.ax_normal.set_ylim(self.hsi_data.min(), 1)
        
        self.canvas_normal = FigureCanvasTkAgg(self.fig_normal, master=normal_frame)
        self.canvas_normal.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Anomalous Spectrum (bottom right)
        anomalous_frame = ttk.LabelFrame(right_frame, text="Anomalous Spectrum")
        anomalous_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        self.fig_anomalous = plt.Figure(figsize=(5, 2.5), dpi=100)  # Same size as normal
        self.ax_anomalous = self.fig_anomalous.add_subplot(111)
        self.ax_anomalous.set_xlabel("Wavelength Band")
        self.ax_anomalous.set_ylabel("Reflectance")
        self.line_anomalous, = self.ax_anomalous.plot([], [], 'r-', alpha=0.7)
        self.ax_anomalous.set_xlim(0, self.hsi_data.shape[2])
        self.ax_anomalous.set_ylim(self.hsi_data.min(), 1)
        
        self.canvas_anomalous = FigureCanvasTkAgg(self.fig_anomalous, master=anomalous_frame)
        self.canvas_anomalous.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Comparison Display (bottom row)
        compare_frame = ttk.LabelFrame(main_frame, text="Spectrum Comparison")
        compare_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        self.fig_compare = plt.Figure(figsize=(10, 3), dpi=100)
        self.ax_compare = self.fig_compare.add_subplot(111)
        self.ax_compare.set_xlabel("Wavelength Band")
        self.ax_compare.set_ylabel("Reflectance")
        self.line_compare_normal, = self.ax_compare.plot([], [], 'g-', label='Normal', alpha=0.7)
        self.line_compare_anomalous, = self.ax_compare.plot([], [], 'r-', label='Anomalous', alpha=0.7)
        self.diff_fill = None
        self.ax_compare.set_xlim(0, self.hsi_data.shape[2])
        self.ax_compare.set_ylim(self.hsi_data.min(), 1)
        self.ax_compare.legend()
        
        self.canvas_compare = FigureCanvasTkAgg(self.fig_compare, master=compare_frame)
        self.canvas_compare.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Controls
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        self.clear_btn = ttk.Button(control_frame, text="Clear Points", command=self.clear_points)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
    
    def on_rgb_click(self, event):
        if event.inaxes != self.ax_rgb:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        
        # Clear previous markers
        for artist in self.ax_rgb.collections:
            artist.remove()
        
        if self.normal_point is None:
            # First click - normal point (green)
            self.normal_point = (x, y)
            self.ax_rgb.scatter(x, y, c='green', s=100, edgecolors='white', linewidths=1.5)
            self.spectrum_history['normal'].append(self.hsi_data[y, x, :])
            self.animate_spectrum('normal')
        elif self.anomalous_point is None:
            # Second click - anomalous point (red)
            self.anomalous_point = (x, y)
            self.ax_rgb.scatter(x, y, c='red', s=100, edgecolors='white', linewidths=1.5)
            self.spectrum_history['anomalous'].append(self.hsi_data[y, x, :])
            self.animate_spectrum('anomalous')
            
            # After both points are selected, animate comparison
            self.animate_comparison()
        
        self.canvas_rgb.draw()
    
    def animate_spectrum(self, spectrum_type):
        """Animate a single spectrum plot with smooth transition"""
        if spectrum_type not in ['normal', 'anomalous']:
            return
            
        current_spectrum = self.spectrum_history[spectrum_type][-1]
        x_data = np.arange(len(current_spectrum))
        
        if spectrum_type == 'normal':
            line = self.line_normal
            canvas = self.canvas_normal
            color = 'g'
        else:
            line = self.line_anomalous
            canvas = self.canvas_anomalous
            color = 'r'
        
        # Get current line data (empty if first time)
        current_line_data = line.get_ydata()
        if len(current_line_data) == 0:
            current_line_data = np.zeros_like(current_spectrum)
        
        # Create interpolation from current to new spectrum
        for i in range(1, self.animation_frames + 1):
            alpha = i / self.animation_frames
            interpolated = current_line_data * (1 - alpha) + current_spectrum * alpha
            line.set_data(x_data, interpolated)
            canvas.draw()
            self.root.update()
            self.root.after(self.animation_interval)
        
        # Ensure final state is exact
        line.set_data(x_data, current_spectrum)
        canvas.draw()
    
    def animate_comparison(self):
        """Animate the comparison plot with both spectra and difference"""
        if not self.normal_point or not self.anomalous_point:
            return
            
        normal_spectrum = self.spectrum_history['normal'][-1]
        anomalous_spectrum = self.spectrum_history['anomalous'][-1]
        x_data = np.arange(len(normal_spectrum))
        
        # Get current line data
        current_normal = self.line_compare_normal.get_ydata()
        current_anomalous = self.line_compare_anomalous.get_ydata()
        
        if len(current_normal) == 0:
            current_normal = np.zeros_like(normal_spectrum)
            current_anomalous = np.zeros_like(anomalous_spectrum)
        
        # Clear previous difference fill
        if self.diff_fill:
            for coll in self.ax_compare.collections:
                if coll == self.diff_fill:
                    coll.remove()
        
        # Animate transition
        for i in range(1, self.animation_frames + 1):
            alpha = i / self.animation_frames
            interp_normal = current_normal * (1 - alpha) + normal_spectrum * alpha
            interp_anomalous = current_anomalous * (1 - alpha) + anomalous_spectrum * alpha
            
            self.line_compare_normal.set_data(x_data, interp_normal)
            self.line_compare_anomalous.set_data(x_data, interp_anomalous)
            
            # Update difference area
            self.diff_fill = self.ax_compare.fill_between(
                x_data, interp_normal, interp_anomalous,
                where=interp_normal < interp_anomalous,
                color='yellow', alpha=0.3
            )
            
            self.canvas_compare.draw()
            self.root.update()
            self.root.after(self.animation_interval)
        
        # Set final state
        self.line_compare_normal.set_data(x_data, normal_spectrum)
        self.line_compare_anomalous.set_data(x_data, anomalous_spectrum)
        self.diff_fill = self.ax_compare.fill_between(
            x_data, normal_spectrum, anomalous_spectrum,
            where=normal_spectrum < anomalous_spectrum,
            color='yellow', alpha=0.3
        )
        self.canvas_compare.draw()
    
    def clear_points(self):
        self.normal_point = None
        self.anomalous_point = None
        self.spectrum_history = {'normal': [], 'anomalous': []}
        
        # Clear plots
        self.line_normal.set_data([], [])
        self.line_anomalous.set_data([], [])
        self.line_compare_normal.set_data([], [])
        self.line_compare_anomalous.set_data([], [])
        
        # Clear markers and difference area
        for artist in self.ax_rgb.collections:
            artist.remove()
        for coll in self.ax_compare.collections:
            coll.remove()
        
        # Redraw all canvases
        self.canvas_rgb.draw()
        self.canvas_normal.draw()
        self.canvas_anomalous.draw()
        self.canvas_compare.draw()

if __name__ == "__main__":
    # Example data - replace with your actual data loading
    hsi_data = np.load("/workspace/src/Session1/cropped/s1_anorm5_bbox_1.npy")  # 100x100 image with 50 bands
    rgb_image = Image.open("/workspace/src/Session1/cropeed_rgbs/Dry_split/s1_anorm5_bbox_1.png")
    rgb_image = np.array(rgb_image)
    root = tk.Tk()
    app = HSIAnalysisApp(root, hsi_data, rgb_image)
    root.geometry("1200x900")
    root.mainloop()