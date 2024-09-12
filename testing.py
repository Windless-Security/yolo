import tkinter as tk
from tkinter import filedialog, ttk
import subprocess
import json
import matplotlib.pyplot as plt
import os
import sys

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
    "Faster R-CNN": ["--video_source", "", "--log_file", log_files["Faster R-CNN"]],
    "SSD": ["--video_source", "", "--log_file", log_files["SSD"]],
    "YOLOv10": ["--no_gui", "--video_source", "", "--log_file", log_files["YOLOv10"]]
}

# Function to run each model script
def run_model(script_name, log_file, args):
    """Run the model script using the virtual environment Python interpreter and log the results."""
    print(f"Running {script_name}...")

    process = subprocess.Popen(
        [venv_python, script_name] + args,  # Use venv's Python with additional arguments
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    for stdout_line in iter(process.stdout.readline, ""):
        print(stdout_line.strip())

    for stderr_line in iter(process.stderr.readline, ""):
        print(stderr_line.strip(), file=sys.stderr)

    process.stdout.close()
    process.stderr.close()

    return_code = process.wait()
    if return_code != 0:
        print(f"Error: {script_name} exited with return code {return_code}")
        return None

    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Error: Log file {log_file} not found for {script_name}!")
        return None

# Function to extract person detection confidence and frame number from the log
def extract_person_detections(log_data, person_class_yolo=0, person_class_coco=1):
    """Extract person detection confidences and frame numbers from the log data."""
    person_detections = {}
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
    return person_detections

def extract_inference_times(log_data):
    """Extract inference times and frame numbers from the log data."""
    inference_times = {}
    for entry in log_data:
        frame_number = entry.get("frame_number")
        inference_time = entry.get("inference_time")
        if frame_number is not None and inference_time is not None:
            inference_times[frame_number] = inference_time
    return inference_times

# Update the plot function
def plot_inference_times(detection_data):
    """Plot inference times for each model."""
    fig, ax = plt.subplots()

    # Extract frame numbers and inference times for each model
    frames = sorted(set().union(*[detection_data[model].keys() for model in detection_data]))
    model_names = list(detection_data.keys())

    for model in model_names:
        times = [detection_data[model].get(frame, 0.0) for frame in frames]
        ax.plot(frames, times, marker='o', label=model)

    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Inference Time (seconds)')
    ax.set_title('Inference Time Comparison Across Models')
    ax.legend()
    plt.show()  

# GUI class
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

        # Result checkboxes for end results
        self.show_accuracy = tk.BooleanVar(value=True)
        self.show_inference_time = tk.BooleanVar(value=False)

        # Chart type options
        self.chart_type = tk.StringVar(value="Bar Chart")
        chart_types = ["Bar Chart", "Line Graph", "Scatter Plot"]

        # Model selection checkboxes
        ttk.Label(root, text="Select Models to Run:").pack(anchor="w", padx=10)
        for model_name in self.model_vars:
            ttk.Checkbutton(root, text=model_name, variable=self.model_vars[model_name]).pack(anchor="w", padx=20)

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

        # Start button
        ttk.Button(root, text="Start Detection", command=self.start_detection).pack(pady=20)

    def browse_file(self):
        """Open a file dialog to select a video file."""
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
        if file_path:
            self.video_path.set(file_path)

    def start_detection(self):
        """Run selected models with the chosen video file."""
        selected_video = self.video_path.get()
        if not os.path.isfile(selected_video):
            print("Invalid video file path.")
            return

        detection_data = {}

        # Run selected models
        for model_name, selected in self.model_vars.items():
            if selected.get():
                log_file = log_files[model_name]
                args = script_args[model_name]

                # Update the video source argument
                args[args.index("--video_source") + 1] = selected_video

                print(f"Running {model_name} on {selected_video}")
                log_data = run_model(model_scripts[model_name], log_file, args)

                if log_data is not None:
                    person_detections = extract_person_detections(log_data)
                    detection_data[model_name] = person_detections
                    print(f"{model_name} person detections: {person_detections}")
                else:
                    print(f"Failed to run {model_name} or log person detections.")

        # Plot based on user selections
        if detection_data:
            self.plot_results(detection_data)

    def plot_results(self, detection_data):
        """Plot the results based on user-selected chart type and results."""
        chart_type = self.chart_type.get()

        # Extract frame numbers and confidences for each model
        frames = sorted(set().union(*[detection_data[model].keys() for model in detection_data]))
        model_names = list(detection_data.keys())

        fig, ax = plt.subplots()

        if chart_type == "Bar Chart":
            self.plot_barchart(ax, frames, model_names, detection_data)
        elif chart_type == "Line Graph":
            self.plot_linegraph(ax, frames, model_names, detection_data)
        elif chart_type == "Scatter Plot":
            self.plot_scatterplot(ax, frames, model_names, detection_data)

        ax.set_xlabel('Frame Number')
        ax.set_title(f'{chart_type} of Detection Confidence')
        ax.legend()
        plt.show()

        # Plot inference times if selected
        if self.show_inference_time.get():
            self.plot_inference_times(detection_data)

    def plot_inference_times(self, detection_data):
        """Wrapper for plotting inference times."""
        inference_data = {}
        for model_name, log_data in detection_data.items():
            inference_data[model_name] = extract_inference_times(log_data)
        
        if inference_data:
            plot_inference_times(inference_data)

    def plot_barchart(self, ax, frames, model_names, detection_data):
        """Plot the results as a bar chart."""
        bar_width = 0.2
        for i, model in enumerate(model_names):
            confidences = [detection_data[model].get(frame, 0.0) for frame in frames]
            ax.bar([frame + i * bar_width for frame in frames], confidences, width=bar_width, label=model)

    def plot_linegraph(self, ax, frames, model_names, detection_data):
        """Plot the results as a line graph."""
        for model in model_names:
            confidences = [detection_data[model].get(frame, 0.0) for frame in frames]
            ax.plot(frames, confidences, marker='o', label=model)

    def plot_scatterplot(self, ax, frames, model_names, detection_data):
        """Plot the results as a scatter plot."""
        for model in model_names:
            confidences = [detection_data[model].get(frame, 0.0) for frame in frames]
            ax.scatter(frames, confidences, label=model)

# Main execution with GUI
def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
