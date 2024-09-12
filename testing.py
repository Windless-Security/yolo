import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import subprocess
import json
import matplotlib.pyplot as plt
import os
import cv2

# Enable interactive plots
plt.ion()

# Define the path to the Python interpreter in the virtual environment
venv_python = r"C:\Users\Sepgi\Desktop\yolo\Scripts\python.exe"  # Modify this to your venv's path

# Define the model scripts to run
model_scripts = {
    "Faster R-CNN": "cnn.py",
    "SSD": "ssd.py",
    "YOLOv10": "vision.py"
}

# Define the result log file for each script
log_files = {
    "Faster R-CNN": "faster_rcnn_log.json",
    "SSD": "ssd_log.json",
    "YOLOv10": "yolo_log.json"
}

# Define additional arguments for each script
script_args = {
    "Faster R-CNN": ["--video_source", "", "--log_file", log_files["Faster R-CNN"], "--threshold", "0.5"],
    "SSD": ["--video_source", "", "--log_file", log_files["SSD"], "--threshold", "0.5"],
    "YOLOv10": ["--no_gui", "--video_source", "", "--log_file", log_files["YOLOv10"], "--conf_threshold", "0.5"]
}

# Function to run each model script with enhanced error handling
def run_model(script_name, log_file, args, progress_callback=None):
    """Run the model script using the virtual environment Python interpreter and log the results."""
    try:
        print(f"Running {script_name} with arguments: {[venv_python, script_name] + args}")
        process = subprocess.Popen(
            [venv_python, script_name] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Capture and log stdout and stderr in real time
        stdout, stderr = process.communicate()

        print(f"Stdout for {script_name}:\n{stdout}")
        if stderr:
            print(f"Error (stderr) for {script_name}:\n{stderr}")

        return_code = process.returncode
        if return_code != 0:
            print(f"Error: {script_name} exited with return code {return_code}")
            print(f"Full error details (stderr):\n{stderr}")
            return None

        # Check if the log file was created successfully
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                return json.load(f)
        else:
            print(f"Error: Log file {log_file} not found for {script_name}!")
            return None

    except Exception as e:
        print(f"Exception occurred while running {script_name}: {str(e)}")
        return None

def visualize_detections(image, boxes, labels, scores):
    """
    Draw bounding boxes and labels on the image.
    """
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush'
    ]

    for i, box in enumerate(boxes):
        if i >= len(labels):
            print(f"Warning: No label for box {i}. Skipping.")
            continue

        label = COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
        score = scores[i]
        x1, y1, x2, y2 = map(int, box)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put label and confidence score
        label_text = f"{label}: {score:.2f}"
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Function to extract person detection confidence and frame number from the log
def extract_person_detections(log_data, person_class_yolo=0, person_class_coco=1):
    """Extract person detection confidences and frame numbers from the log data."""
    person_detections = {}
    try:
        for entry in log_data:
            frame_number = entry.get("frame_number")
            if frame_number is None:
                continue
            detected = False
            for label, score in zip(entry["labels"], entry["scores"]):
                if label == person_class_yolo or label == person_class_coco:
                    person_detections[frame_number] = score
                    detected = True
                    break
            if not detected:
                pass  # Skip frames without person detection
    except KeyError as e:
        print(f"Error processing detection data: missing key {e}")
    except Exception as e:
        print(f"Unexpected error while extracting person detections: {str(e)}")
    return person_detections

# Function to extract inference times
def extract_inference_times(log_data):
    """Extract inference times and frame numbers from the log data."""
    inference_times = {}
    try:
        for entry in log_data:
            if isinstance(entry, dict):
                frame_number = entry.get("frame_number")
                inference_time = entry.get("inference_time")
                if frame_number is not None and inference_time is not None:
                    inference_times[frame_number] = inference_time
            else:
                print(f"Warning: Invalid log entry format: {entry}")
    except KeyError as e:
        print(f"Error processing inference time data: missing key {e}")
    except Exception as e:
        print(f"Unexpected error while extracting inference times: {str(e)}")
    return inference_times

# Update the plot function with error handling
def plot_inference_times(detection_data):
    """Plot inference times for each model."""
    try:
        fig, ax = plt.subplots()
        for model in detection_data:
            times = detection_data[model]
            frames = sorted(times.keys())
            inference_times = [times[frame] for frame in frames]
            ax.plot(frames, inference_times, marker='o', label=model)

        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Inference Time (seconds)')
        ax.set_title('Inference Time Comparison Across Models')
        ax.legend()
        plt.show()
    except Exception as e:
        print(f"Error plotting inference times: {str(e)}")

# GUI class
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Selection and Video Source")

        # Variables for checkboxes, dropdowns, and video path
        self.model_vars = {
            "Faster R-CNN": tk.BooleanVar(),
            "SSD": tk.BooleanVar(),
            "YOLOv10": tk.BooleanVar(),
        }
        self.video_path = tk.StringVar()

        # Threshold variables
        self.threshold_vars = {
            "Faster R-CNN": tk.DoubleVar(value=0.5),
            "SSD": tk.DoubleVar(value=0.5),
            "YOLOv10": tk.DoubleVar(value=0.5),
        }

        # Result checkboxes for end results
        self.show_accuracy = tk.BooleanVar(value=True)
        self.show_inference_time = tk.BooleanVar(value=False)

        # Chart type options
        self.chart_type = tk.StringVar(value="Bar Chart")
        chart_types = ["Bar Chart", "Line Graph", "Scatter Plot"]

        # Progress variable
        self.progress = tk.DoubleVar()

        # Model selection checkboxes
        ttk.Label(root, text="Select Models to Run:").pack(anchor="w", padx=10)
        for model_name in self.model_vars:
            frame = ttk.Frame(root)
            frame.pack(anchor="w", padx=20)
            ttk.Checkbutton(frame, text=model_name, variable=self.model_vars[model_name]).pack(side="left")
            ttk.Label(frame, text="Threshold:").pack(side="left", padx=5)
            ttk.Scale(frame, from_=0.0, to=1.0, orient="horizontal", variable=self.threshold_vars[model_name], length=150).pack(side="left")

        # Video file path selection
        ttk.Label(root, text="Select Video File:").pack(anchor="w", padx=10)
        self.path_entry = ttk.Entry(root, textvariable=self.video_path)
        self.path_entry.pack(padx=10, pady=5, fill="x")
        self.browse_button = ttk.Button(root, text="Browse", command=self.browse_file)
        self.browse_button.pack(pady=5)

        # Dropdown for chart type selection
        ttk.Label(root, text="Select Chart Type:").pack(anchor="w", padx=10)
        self.chart_dropdown = ttk.Combobox(root, textvariable=self.chart_type, values=chart_types, state="readonly")
        self.chart_dropdown.pack(padx=10, pady=5, fill="x")

        # Checkboxes for what results to show
        ttk.Label(root, text="Select Results to Display:").pack(anchor="w", padx=10)
        ttk.Checkbutton(root, text="Person Detection Accuracy", variable=self.show_accuracy).pack(anchor="w", padx=20)
        ttk.Checkbutton(root, text="Inference Time", variable=self.show_inference_time).pack(anchor="w", padx=20)

        # Progress Bar
        self.progress_bar = ttk.Progressbar(root, variable=self.progress, maximum=100)
        self.progress_bar.pack(padx=10, pady=10, fill="x")

        # Start button
        ttk.Button(root, text="Start Detection", command=self.start_detection).pack(pady=20)

    def browse_file(self):
        """Open a file dialog to select a video file."""
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
            if file_path:
                self.video_path.set(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Error browsing file: {str(e)}")

    def start_detection(self):
        """Run selected models with the chosen video file."""
        selected_video = self.video_path.get()
        if not os.path.isfile(selected_video):
            messagebox.showerror("Error", "Invalid video file path.")
            return

        detection_data = {}
        total_models = sum(1 for selected in self.model_vars.values() if selected.get())
        current_model = 0

        # Run selected models
        for model_name, selected in self.model_vars.items():
            if selected.get():
                log_file = log_files[model_name]
                args = script_args[model_name].copy()

                # Update the video source argument
                args[args.index("--video_source") + 1] = selected_video

                # Update the threshold argument
                threshold = str(self.threshold_vars[model_name].get())
                if "--threshold" in args:
                    args[args.index("--threshold") + 1] = threshold
                elif "--conf_threshold" in args:
                    args[args.index("--conf_threshold") + 1] = threshold

                print(f"Running {model_name} on {selected_video}")

                # Update progress
                current_model += 1
                progress_percent = (current_model / total_models) * 100
                self.progress.set(progress_percent)
                self.root.update_idletasks()

                log_data = run_model(model_scripts[model_name], log_file, args)

                if log_data is not None:
                    person_detections = extract_person_detections(log_data)
                    detection_data[model_name] = {
                        'person_detections': person_detections,
                        'log_data': log_data
                    }
                    print(f"{model_name} person detections: {person_detections}")
                else:
                    print(f"Failed to run {model_name} or log person detections.")
                    messagebox.showerror("Error", f"Failed to run {model_name} or log person detections.")

        # Plot based on user selections
        if detection_data:
            self.plot_results(detection_data)
        else:
            messagebox.showinfo("Info", "No data to display.")

        # Reset progress
        self.progress.set(0)
        self.root.update_idletasks()

    def plot_results(self, detection_data):
        """Plot the results based on user-selected chart type and results."""
        if self.show_accuracy.get():
            chart_type = self.chart_type.get()

            # Extract frame numbers and confidences for each model
            frames = sorted(set().union(*[detection_data[model]['person_detections'].keys() for model in detection_data]))
            model_names = list(detection_data.keys())

            fig, ax = plt.subplots()

            if chart_type == "Bar Chart":
                self.plot_barchart(ax, frames, model_names, detection_data)
            elif chart_type == "Line Graph":
                self.plot_linegraph(ax, frames, model_names, detection_data)
            elif chart_type == "Scatter Plot":
                self.plot_scatterplot(ax, frames, model_names, detection_data)

            ax.set_xlabel('Frame Number')
            ax.set_ylabel('Detection Confidence')
            ax.set_title(f'{chart_type} of Detection Confidence')
            ax.legend()

            # Enable zoom and pan
            plt.tight_layout()
            plt.show()

        # Plot inference times if selected
        if self.show_inference_time.get():
            self.plot_inference_times(detection_data)

    def plot_inference_times(self, detection_data):
        """Wrapper for plotting inference times."""
        inference_data = {}
        for model_name, data in detection_data.items():
            log_data = data['log_data']
            inference_times = extract_inference_times(log_data)
            inference_data[model_name] = inference_times

        if inference_data:
            plot_inference_times(inference_data)

    def plot_barchart(self, ax, frames, model_names, detection_data):
        """Plot the results as a bar chart."""
        try:
            bar_width = 0.2
            for i, model in enumerate(model_names):
                confidences = [detection_data[model]['person_detections'].get(frame, 0.0) for frame in frames]
                ax.bar([frame + i * bar_width for frame in frames], confidences, width=bar_width, label=model)
        except Exception as e:
            print(f"Error plotting bar chart: {str(e)}")

    def plot_linegraph(self, ax, frames, model_names, detection_data):
        """Plot the results as a line graph."""
        try:
            for model in model_names:
                confidences = [detection_data[model]['person_detections'].get(frame, 0.0) for frame in frames]
                ax.plot(frames, confidences, marker='o', label=model)
        except Exception as e:
            print(f"Error plotting line graph: {str(e)}")

    def plot_scatterplot(self, ax, frames, model_names, detection_data):
        """Plot the results as a scatter plot."""
        try:
            for model in model_names:
                confidences = [detection_data[model]['person_detections'].get(frame, 0.0) for frame in frames]
                ax.scatter(frames, confidences, label=model)
        except Exception as e:
            print(f"Error plotting scatter plot: {str(e)}")

# Main execution with GUI
def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
