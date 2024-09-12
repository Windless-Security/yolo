import cv2
import os
import json
import argparse
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from ultralytics import YOLO
import time
from collections import defaultdict
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import logging
import psutil
import platform
import numpy as np

try:
    import GPUtil  # For GPU usage
except ImportError:
    GPUtil = None  # Handle absence of GPUtil

# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Set tracker type
tracker_type = 'CSRT'

def create_tracker(tracker_type='CSRT'):
    """Create object tracker based on OpenCV version and tracker type."""
    major_ver = cv2.__version__.split('.')[0]
    if int(major_ver) < 4:
        # OpenCV 3.x
        if tracker_type == 'CSRT':
            tracker = cv2.TrackerCSRT_create()
        elif tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        else:
            tracker = cv2.TrackerMOSSE_create()
    else:
        # OpenCV 4.x and above
        if hasattr(cv2, 'legacy'):
            if tracker_type == 'CSRT':
                tracker = cv2.legacy.TrackerCSRT_create()
            elif tracker_type == 'KCF':
                tracker = cv2.legacy.TrackerKCF_create()
            else:
                tracker = cv2.legacy.TrackerMOSSE_create()
        elif cv2.__version__.startswith('4'):
            # For OpenCV 4.x versions without 'legacy' module
            if tracker_type == 'CSRT':
                tracker = cv2.TrackerCSRT_create()
            elif tracker_type == 'KCF':
                tracker = cv2.TrackerKCF_create()
            else:
                tracker = cv2.TrackerMOSSE_create()
        else:
            # If no appropriate tracker is found
            tracker = None
    return tracker

def create_multitracker():
    """Create MultiTracker object based on OpenCV version."""
    if hasattr(cv2, 'legacy'):
        multi_tracker = cv2.legacy.MultiTracker_create()
    elif hasattr(cv2, 'MultiTracker_create'):
        multi_tracker = cv2.MultiTracker_create()
    else:
        # For OpenCV versions without MultiTracker support
        multi_tracker = None
    return multi_tracker

def list_available_cameras():
    available_cameras = []
    for i in range(10):  # Check the first 10 cameras
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def list_pt_models(directory="."):
    """List all available .pt models in the given directory."""
    return [f for f in os.listdir(directory) if f.endswith(".pt")]

def log_detections(detection_data, log_file="detections_log.json"):
    """Save the detection data to a JSON file."""
    with open(log_file, "w") as outfile:
        json.dump(detection_data, outfile, indent=4)
    print(f"Detection log saved to {log_file}")

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv10 Object Detection")

        # Variables
        self.selected_model = tk.StringVar()
        self.fps_limit = tk.StringVar(value="30")
        self.input_mode = tk.StringVar(value="camera")
        self.video_path = tk.StringVar()
        self.selected_camera = tk.IntVar(value=-1)
        self.conf_threshold = tk.DoubleVar(value=0.5)
        self.nms_threshold = tk.DoubleVar(value=0.5)
        self.save_video = tk.BooleanVar()
        self.progress = tk.DoubleVar()
        self.input_size = tk.IntVar(value=640)
        self.paused = tk.BooleanVar(value=False)
        self.is_running = False  # Flag to control the processing loop
        self.enable_tracking = tk.BooleanVar(value=True)  # Default is tracking enabled

        # Main frame
        main_frame = ttk.Frame(root)
        main_frame.pack(fill='both', expand=True)

        # Configure grid
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # Header
        ttk.Label(main_frame, text="YOLOv10 Object Detection", font=("Arial", 16)).grid(row=0, column=0, pady=10)

        # Model selection
        ttk.Label(main_frame, text="Select Model:").grid(row=1, column=0, sticky='w', padx=10)
        model_frame = ttk.Frame(main_frame)
        model_frame.grid(row=2, column=0, sticky='we', padx=10)
        model_frame.columnconfigure(0, weight=1)
        self.model_dropdown = ttk.Combobox(model_frame, textvariable=self.selected_model, values=list_pt_models(), state="readonly")
        self.model_dropdown.grid(row=0, column=0, sticky='we')
        self.model_dropdown.set("yolov8n.pt")  # Default model
        ttk.Button(model_frame, text="Browse Model", command=self.browse_model).grid(row=0, column=1, padx=5)
        self.model_dropdown.bind("<Enter>", lambda e: self.show_tooltip(e, "Select the YOLO model to use."))

        # Input Mode
        ttk.Label(main_frame, text="Input Mode:").grid(row=3, column=0, sticky='w', padx=10)
        input_mode_frame = ttk.Frame(main_frame)
        input_mode_frame.grid(row=4, column=0, sticky='we', padx=20)
        ttk.Radiobutton(input_mode_frame, text="Use Camera", variable=self.input_mode, value="camera").grid(row=0, column=0)
        ttk.Radiobutton(input_mode_frame, text="Use Video File", variable=self.input_mode, value="video").grid(row=0, column=1)

        # Camera selection
        self.camera_dropdown = ttk.Combobox(main_frame, textvariable=self.selected_camera, state="readonly")
        self.update_camera_list()
        self.camera_dropdown.grid(row=5, column=0, sticky='we', padx=10, pady=5)

        # Video file path
        ttk.Label(main_frame, text="Video File Path:").grid(row=6, column=0, sticky='w', padx=10)
        video_path_frame = ttk.Frame(main_frame)
        video_path_frame.grid(row=7, column=0, sticky='we', padx=10)
        video_path_frame.columnconfigure(0, weight=1)
        self.path_entry = ttk.Entry(video_path_frame, textvariable=self.video_path, state="disabled")
        self.path_entry.grid(row=0, column=0, sticky='we')
        self.browse_button = ttk.Button(video_path_frame, text="Browse", command=self.browse_file, state="disabled")
        self.browse_button.grid(row=0, column=1, padx=5)

        # Confidence Threshold Slider
        ttk.Label(main_frame, text="Confidence Threshold:").grid(row=8, column=0, sticky='w', padx=10)
        self.conf_slider = ttk.Scale(main_frame, from_=0.0, to=1.0, orient="horizontal", variable=self.conf_threshold)
        self.conf_slider.grid(row=9, column=0, sticky='we', padx=10, pady=5)
        self.conf_slider.bind("<Enter>", lambda e: self.show_tooltip(e, "Adjust the minimum confidence for detections."))

        # NMS Threshold Slider
        ttk.Label(main_frame, text="NMS Threshold:").grid(row=10, column=0, sticky='w', padx=10)
        self.nms_slider = ttk.Scale(main_frame, from_=0.0, to=1.0, orient="horizontal", variable=self.nms_threshold)
        self.nms_slider.grid(row=11, column=0, sticky='we', padx=10, pady=5)
        self.nms_slider.bind("<Enter>", lambda e: self.show_tooltip(e, "Adjust the Non-Maximum Suppression threshold."))

        # Input Size Dropdown
        ttk.Label(main_frame, text="Input Size:").grid(row=12, column=0, sticky='w', padx=10)
        input_sizes = [320, 480, 640, 800, 960]
        self.input_size_dropdown = ttk.Combobox(main_frame, textvariable=self.input_size, values=input_sizes, state="readonly")
        self.input_size_dropdown.grid(row=13, column=0, sticky='we', padx=10, pady=5)

        # Save Video Checkbox
        self.save_video_checkbox = ttk.Checkbutton(main_frame, text="Save Annotated Video", variable=self.save_video)
        self.save_video_checkbox.grid(row=14, column=0, sticky='w', padx=10)
        self.save_video_checkbox.bind("<Enter>", lambda e: self.show_tooltip(e, "Enable to save the output video with annotations."))

        # Enable Object Tracking Checkbox
        self.tracking_checkbox = ttk.Checkbutton(main_frame, text="Enable Object Tracking", variable=self.enable_tracking)
        self.tracking_checkbox.grid(row=15, column=0, sticky='w', padx=10)
        self.tracking_checkbox.bind("<Enter>", lambda e: self.show_tooltip(e, "Enable or disable object tracking."))

        # FPS Limit Dropdown
        ttk.Label(main_frame, text="FPS Limit:").grid(row=16, column=0, sticky='w', padx=10)
        self.fps_dropdown = ttk.Combobox(main_frame, textvariable=self.fps_limit, values=["0", "15", "30", "60"], state="readonly")
        self.fps_dropdown.grid(row=17, column=0, sticky='we', padx=10, pady=5)
        self.fps_dropdown.bind("<Enter>", lambda e: self.show_tooltip(e, "Set the maximum FPS for processing. 0 for no limit."))

        # Progress Bar
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress, maximum=100)
        self.progress_bar.grid(row=18, column=0, sticky='we', padx=10, pady=10)

        # Control Buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=19, column=0, pady=10)
        ttk.Button(control_frame, text="Start Detection", command=self.start_detection).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Pause", command=self.pause_detection).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Resume", command=self.resume_detection).grid(row=0, column=2, padx=5)

        # Save and Load Settings
        settings_frame = ttk.Frame(main_frame)
        settings_frame.grid(row=20, column=0, pady=5)
        ttk.Button(settings_frame, text="Save Settings", command=self.save_settings).grid(row=0, column=0, padx=5)
        ttk.Button(settings_frame, text="Load Settings", command=self.load_settings).grid(row=0, column=1, padx=5)

        # Plot Frame
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=21, column=0, sticky='nsew', padx=10, pady=10)
        main_frame.rowconfigure(21, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        # Create the figure and canvas
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'r-')
        self.ax.set_xlabel('Frame Number')
        self.ax.set_ylabel('Inference Time (seconds)')
        self.ax.set_title('Live Inference Time')
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Binding input mode changes
        self.input_mode.trace("w", self.toggle_input_mode)

    def show_tooltip(self, event, text):
        """Display a tooltip with the given text."""
        x = event.widget.winfo_rootx() + 20
        y = event.widget.winfo_rooty() + 20
        self.tooltip = tk.Toplevel(self.root)
        self.tooltip.overrideredirect(True)
        self.tooltip.geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=text, background="yellow", relief="solid", borderwidth=1)
        label.pack()
        event.widget.bind("<Leave>", self.hide_tooltip)

    def hide_tooltip(self, event):
        """Hide the tooltip."""
        if hasattr(self, 'tooltip'):
            self.tooltip.destroy()

    def update_camera_list(self):
        """Update the available camera list dynamically."""
        cameras = list_available_cameras()
        if cameras:
            self.camera_dropdown["values"] = cameras
            self.camera_dropdown.set(cameras[0])
        else:
            self.camera_dropdown["values"] = ["No cameras found"]

    def browse_file(self):
        """Open a file dialog to select a video file."""
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
        if file_path:
            self.video_path.set(file_path)

    def toggle_input_mode(self, *args):
        """Enable/Disable fields based on input mode."""
        if self.input_mode.get() == "camera":
            self.path_entry.config(state="disabled")
            self.browse_button.config(state="disabled")
            self.camera_dropdown.config(state="readonly")
        else:
            self.path_entry.config(state="normal")
            self.browse_button.config(state="normal")
            self.camera_dropdown.config(state="disabled")

    def browse_model(self):
        """Open a file dialog to select a custom model file."""
        file_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pt")])
        if file_path:
            self.selected_model.set(file_path)

    def start_detection(self):
        """Start object detection with the selected options."""
        if self.is_running:
            messagebox.showinfo("Info", "Detection is already running.")
            return

        selected_model = self.selected_model.get()
        fps = int(self.fps_limit.get())
        conf_thresh = self.conf_threshold.get()
        nms_thresh = self.nms_threshold.get()
        save_video = self.save_video.get()
        input_size = self.input_size.get()

        log_file = "detections_log.json"  # JSON log file name

        try:
            # Display progress indicator
            self.progress.set(10)
            self.root.update_idletasks()

            # Load the model (this may take some time)
            self.model = YOLO(selected_model)

            self.progress.set(50)
            self.root.update_idletasks()

            if self.input_mode.get() == "camera":
                camera_index = self.selected_camera.get()
                input_source = int(camera_index)
            elif self.input_mode.get() == "video":
                video_path = self.video_path.get()
                if os.path.isfile(video_path):
                    input_source = video_path
                else:
                    messagebox.showerror("Error", "Invalid video file path")
                    self.progress.set(0)
                    return
            else:
                messagebox.showerror("Error", "Invalid input mode selected")
                self.progress.set(0)
                return

            # Set up processing variables
            self.input_source = input_source
            self.fps = fps
            self.conf_thresh = conf_thresh
            self.nms_thresh = nms_thresh
            self.save_video_flag = save_video
            self.input_size_value = input_size
            self.log_file = log_file
            self.frame_number = 0
            self.paused_flag = False
            self.is_running = True
            self.enable_tracking_flag = self.enable_tracking.get()

            # Initialize class statistics and data lists
            self.class_stats = defaultdict(lambda: {'count': 0, 'total_confidence': 0.0})
            self.frame_numbers = []
            self.inference_times = []
            self.detection_logs = []

            # Open video capture
            self.cap = cv2.VideoCapture(self.input_source)

            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Could not open video source {self.input_source}")
                self.progress.set(0)
                self.is_running = False
                return

            # Get video properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_video = self.cap.get(cv2.CAP_PROP_FPS) if self.fps == 0 else self.fps

            # Video Writer if saving is enabled
            if self.save_video_flag:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.output_video = cv2.VideoWriter('annotated_output.mp4', fourcc, fps_video,
                                                    (self.frame_width, self.frame_height))

            # Initialize trackers
            self.multi_tracker = create_multitracker()
            if self.multi_tracker is None:
                print("Warning: MultiTracker is not available in this OpenCV version.")
                self.use_tracking = False
            else:
                self.use_tracking = True

            # Initialize variables for FPS calculation
            self.prev_time = time.time()

            # Start processing frames
            self.process_frame()

            self.progress.set(100)
            self.root.update_idletasks()

        except Exception as e:
            logging.error(f"An error occurred during detection: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.progress.set(0)
            self.is_running = False
            self.root.update_idletasks()
            return

    def process_frame(self):
        """Process a single frame and schedule the next frame processing."""
        if not self.is_running or self.paused_flag:
            if self.cap:
                self.cap.release()
            if self.save_video_flag and hasattr(self, 'output_video'):
                self.output_video.release()
            self.is_running = False
            return

        ret, frame = self.cap.read()
        if not ret:
            # End of video
            self.cap.release()
            if self.save_video_flag and hasattr(self, 'output_video'):
                self.output_video.release()
            cv2.destroyAllWindows()
            self.is_running = False

            # Display per-class statistics
            print("\nPer-Class Statistics:")
            for class_name, stats in self.class_stats.items():
                average_confidence = stats['total_confidence'] / stats['count']
                print(f"Class: {class_name}, Count: {stats['count']}, Average Confidence: {average_confidence:.2f}")

            # Save detection logs to JSON
            log_detections(self.detection_logs, self.log_file)
            return

        self.frame_number += 1

        # Record the start time before inference
        start_time = time.time()

        # Resize frame to the input size
        frame_resized = cv2.resize(frame, (self.input_size_value, self.input_size_value))

        # Run YOLO inference
        results = self.model(frame_resized, conf=self.conf_thresh, iou=self.nms_thresh)

        # Record the end time after inference
        inference_time = time.time() - start_time

        # Log detection results
        detections = results[0].boxes
        labels = []  # To store labels (class IDs)
        scores = []  # To store confidence scores
        boxes = []   # To store bounding boxes

        for det in detections:
            confidence = float(det.conf)

            class_id = int(det.cls)
            class_name = self.model.names[class_id]
            bounding_box = det.xyxy[0].tolist()

            # Adjust bounding box coordinates according to original frame size
            x1, y1, x2, y2 = bounding_box
            x1 = int(x1 * frame.shape[1] / self.input_size_value)
            x2 = int(x2 * frame.shape[1] / self.input_size_value)
            y1 = int(y1 * frame.shape[0] / self.input_size_value)
            y2 = int(y2 * frame.shape[0] / self.input_size_value)
            bounding_box = [x1, y1, x2, y2]

            # Add labels, scores, and boxes for logging
            labels.append(class_id)  # Store class IDs (labels)
            scores.append(confidence)  # Store confidence scores
            boxes.append(bounding_box)

            # Update class statistics
            self.class_stats[class_name]['count'] += 1
            self.class_stats[class_name]['total_confidence'] += confidence

        # Annotate frame
        annotated_frame = frame.copy()
        for bbox, label_id, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = bbox
            class_name = self.model.names[label_id]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{class_name}: {score:.2f}"

            # Display the label text
            cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Approximate distance estimation for 'person' class
            if class_name == 'person':
                # Calculate pixel height
                pixel_height = y2 - y1

                # Estimated focal length in pixels (this value may need adjustment)
                focal_length_pixels = 800  # Adjust based on your camera

                # Real-world height of a person in cm
                real_height_cm = 170  # Average human height

                # Estimate distance in cm
                if pixel_height > 0:
                    distance_cm = (real_height_cm * focal_length_pixels) / pixel_height
                else:
                    distance_cm = 0

                # Display the distance on the frame
                cv2.putText(annotated_frame, f"Distance: {distance_cm:.2f} cm", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Update trackers only if tracking is enabled
        if self.enable_tracking_flag and self.use_tracking:
            if self.frame_number == 1:
                # Initialize trackers for detected objects
                for bbox in boxes:
                    x1, y1, x2, y2 = bbox
                    w = x2 - x1
                    h = y2 - y1
                    tracker = create_tracker(tracker_type)
                    if tracker is not None:
                        self.multi_tracker.add(tracker, frame, (x1, y1, w, h))
            else:
                success, tracked_boxes = self.multi_tracker.update(frame)
                if success:
                    for i, new_box in enumerate(tracked_boxes):
                        x, y, w, h = map(int, new_box)
                        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        cv2.putText(annotated_frame, f"ID {i+1}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time

        # Overlay FPS on the frame
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Get CPU and GPU usage
        cpu_usage = psutil.cpu_percent()
        cv2.putText(annotated_frame, f"CPU Usage: {cpu_usage}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # GPU usage
        if GPUtil:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_usage = gpu.load * 100
                cv2.putText(annotated_frame, f"GPU Usage: {gpu_usage:.2f}%", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Append data to lists for plotting
        self.frame_numbers.append(self.frame_number)
        self.inference_times.append(inference_time)

        # Update the live plot
        self.line.set_xdata(self.frame_numbers)
        self.line.set_ydata(self.inference_times)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

        # Display the frame using OpenCV
        cv2.imshow("YOLOv10 Detection", annotated_frame)
        cv2.waitKey(1)  # Necessary for OpenCV window to update

        # Save the frame if saving is enabled
        if self.save_video_flag:
            self.output_video.write(annotated_frame)

        # Log entry for this frame
        log_entry = {
            "frame_number": self.frame_number,
            "inference_time": inference_time,
            "labels": labels,
            "scores": scores,
            "boxes": boxes
        }
        self.detection_logs.append(log_entry)

        # Schedule the next frame processing
        self.root.after(1, self.process_frame)

    def pause_detection(self):
        """Pause the detection process."""
        if not self.is_running:
            return
        self.paused_flag = True
        self.is_running = False

    def resume_detection(self):
        """Resume the detection process."""
        if self.is_running:
            return
        self.paused_flag = False
        self.is_running = True
        # Re-open video capture if necessary
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.input_source)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        self.prev_time = time.time()
        self.process_frame()

    def save_settings(self):
        """Save the current settings to a JSON file."""
        settings = {
            'selected_model': self.selected_model.get(),
            'fps_limit': self.fps_limit.get(),
            'input_mode': self.input_mode.get(),
            'video_path': self.video_path.get(),
            'selected_camera': self.selected_camera.get(),
            'conf_threshold': self.conf_threshold.get(),
            'nms_threshold': self.nms_threshold.get(),
            'save_video': self.save_video.get(),
            'input_size': self.input_size.get(),
            'enable_tracking': self.enable_tracking.get(),
        }
        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(settings, f, indent=4)
            messagebox.showinfo("Settings Saved", f"Settings saved to {file_path}")

    def load_settings(self):
        """Load settings from a JSON file."""
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'r') as f:
                settings = json.load(f)
            # Update GUI elements with loaded settings
            self.selected_model.set(settings.get('selected_model', self.selected_model.get()))
            self.fps_limit.set(settings.get('fps_limit', self.fps_limit.get()))
            self.input_mode.set(settings.get('input_mode', self.input_mode.get()))
            self.video_path.set(settings.get('video_path', self.video_path.get()))
            self.selected_camera.set(settings.get('selected_camera', self.selected_camera.get()))
            self.conf_threshold.set(settings.get('conf_threshold', self.conf_threshold.get()))
            self.nms_threshold.set(settings.get('nms_threshold', self.nms_threshold.get()))
            self.save_video.set(settings.get('save_video', self.save_video.get()))
            self.input_size.set(settings.get('input_size', self.input_size.get()))
            self.enable_tracking.set(settings.get('enable_tracking', self.enable_tracking.get()))
            messagebox.showinfo("Settings Loaded", f"Settings loaded from {file_path}")

def main():
    parser = argparse.ArgumentParser(description="YOLOv10 Object Detection")
    parser.add_argument("--no_gui", action="store_true", help="Run YOLO without GUI and directly via command-line arguments")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model file")
    parser.add_argument("--video_source", type=str, default="0", help="Path to the video file or camera index")
    parser.add_argument("--fps", type=int, default=30, help="FPS limit for the detection")
    parser.add_argument("--log_file", type=str, default="detections_log.json", help="File to save detections log")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--nms_threshold", type=float, default=0.5, help="NMS threshold for detections")
    parser.add_argument("--save_video", action="store_true", help="Save the annotated video output")
    parser.add_argument("--input_size", type=int, default=640, help="Input size for the model")
    parser.add_argument("--use_tensorrt", action="store_true", help="Use TensorRT for inference (requires compatible environment)")
    parser.add_argument("--enable_tracking", action="store_true", help="Enable object tracking")

    args = parser.parse_args()

    if args.no_gui:
        try:
            # Load the model
            if args.use_tensorrt:
                # Load TensorRT model (requires proper setup)
                # Placeholder for TensorRT integration
                model = YOLO(args.model)
                print("TensorRT integration is not implemented in this script.")
            else:
                model = YOLO(args.model)

            # Create a dummy root for OpenCV windows
            root = tk.Tk()
            root.withdraw()

            # Since we're not using the GUI, we need to call the processing function directly
            app = App(root)
            app.model = model
            app.input_source = args.video_source
            app.fps = args.fps
            app.conf_thresh = args.conf_threshold
            app.nms_thresh = args.nms_threshold
            app.save_video_flag = args.save_video
            app.input_size_value = args.input_size
            app.log_file = args.log_file
            app.enable_tracking_flag = args.enable_tracking

            # Initialize variables
            app.frame_number = 0
            app.paused_flag = False
            app.is_running = True

            # Initialize class statistics and data lists
            app.class_stats = defaultdict(lambda: {'count': 0, 'total_confidence': 0.0})
            app.frame_numbers = []
            app.inference_times = []
            app.detection_logs = []

            # Open video capture
            if args.video_source.isdigit():
                app.input_source = int(args.video_source)
            else:
                app.input_source = args.video_source

            app.cap = cv2.VideoCapture(app.input_source)

            if not app.cap.isOpened():
                print(f"Error: Could not open video source {app.input_source}")
                return

            # Get video properties
            app.frame_width = int(app.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            app.frame_height = int(app.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_video = app.cap.get(cv2.CAP_PROP_FPS) if app.fps == 0 else app.fps

            # Video Writer if saving is enabled
            if app.save_video_flag:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                app.output_video = cv2.VideoWriter('annotated_output.mp4', fourcc, fps_video,
                                                   (app.frame_width, app.frame_height))

            # Initialize trackers
            app.multi_tracker = create_multitracker()
            if app.multi_tracker is None:
                print("Warning: MultiTracker is not available in this OpenCV version.")
                app.use_tracking = False
            else:
                app.use_tracking = True

            # Initialize variables for FPS calculation
            app.prev_time = time.time()

            # Start processing frames
            while app.is_running:
                app.process_frame()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    app.is_running = False
                    break

            # Release resources
            app.cap.release()
            if app.save_video_flag and hasattr(app, 'output_video'):
                app.output_video.release()
            cv2.destroyAllWindows()

        except Exception as e:
            logging.error(f"An error occurred during detection: {str(e)}", exc_info=True)
            print(f"An error occurred: {str(e)}")
    else:
        root = tk.Tk()
        app = App(root)
        root.mainloop()

if __name__ == "__main__":
    main()
