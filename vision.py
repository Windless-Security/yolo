import cv2
import os
import json
import argparse
import tkinter as tk
from tkinter import filedialog, ttk
from ultralytics import YOLO
import time
from collections import defaultdict

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


# Modify the logging part in process_video_or_camera for YOLO
def process_video_or_camera(input_source, model, fps_limit=30, log_file="detections_log.json"):
    """Process the video or camera, log detections, and track objects per frame."""
    cap = cv2.VideoCapture(input_source)

    if not cap.isOpened():
        print(f"Error: Could not open video or camera source {input_source}")
        return

    detection_logs = []  # List to store all detection logs

    # Object tracker (to keep track of objects across frames)
    object_tracker = defaultdict(lambda: {"last_seen": None, "last_logged": None})

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get current frame number for logging
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Record the start time before inference
        start_time = time.time()

        # Run YOLO inference
        results = model(frame)

        # Record the end time after inference
        inference_time = time.time() - start_time

        annotated_frame = results[0].plot()

        # Log detection results
        detections = results[0].boxes
        labels = []  # To store labels (class IDs)
        scores = []  # To store confidence scores

        for det in detections:
            confidence = float(det.conf)

            class_id = int(det.cls)
            class_name = model.names[class_id]
            bounding_box = det.xyxy.tolist()

            # Add labels and scores for logging
            labels.append(class_id)  # Store class IDs (labels)
            scores.append(confidence)  # Store confidence scores

            # Create a unique key for this object (based on class and bounding box)
            object_key = f"{class_name}_{bounding_box}"

            # Check if the object is already being tracked
            if object_key in object_tracker:
                # Object is already in the frame, check if we should log it again (every 5 seconds)
                if frame_number - object_tracker[object_key]["last_logged"] >= 5:
                    log_entry = {
                        "frame_number": frame_number,  # Log frame number instead of timestamp
                        "class": class_name,
                        "confidence": confidence,
                        "bounding_box": bounding_box,
                        "inference_time": inference_time,  # Log inference time for this frame
                        "labels": labels,  # Add labels to log
                        "scores": scores  # Add scores to log
                    }
                    detection_logs.append(log_entry)
                    object_tracker[object_key]["last_logged"] = frame_number
            else:
                # New object detected
                log_entry = {
                    "frame_number": frame_number,  # Log frame number instead of timestamp
                    "class": class_name,
                    "confidence": confidence,
                    "bounding_box": bounding_box,
                    "inference_time": inference_time,  # Log inference time for this frame
                    "labels": labels,  # Add labels to log
                    "scores": scores  # Add scores to log
                }
                detection_logs.append(log_entry)
                # Update the tracker for this object
                object_tracker[object_key] = {"last_seen": frame_number, "last_logged": frame_number}

        # Remove objects from the tracker if they haven't been seen for a while
        object_tracker = {
            k: v for k, v in object_tracker.items() if frame_number - v["last_seen"] <= 5
        }

        cv2.imshow("YOLOv10 Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save detection logs to JSON
    log_detections(detection_logs, log_file)


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

        # Header
        ttk.Label(root, text="YOLOv10 Object Detection", font=("Arial", 16)).pack(pady=10)

        # Dropdown for available .pt models
        ttk.Label(root, text="Select Model:").pack(anchor="w", padx=10)
        self.model_dropdown = ttk.Combobox(root, textvariable=self.selected_model, values=list_pt_models(), state="readonly")
        self.model_dropdown.pack(padx=10, pady=5, fill="x")
        self.model_dropdown.set("yolov10x.pt")  # Default model

        # Input Mode (Radio buttons for camera or video)
        ttk.Label(root, text="Input Mode:").pack(anchor="w", padx=10)
        ttk.Radiobutton(root, text="Use Camera", variable=self.input_mode, value="camera").pack(anchor="w", padx=20)
        ttk.Radiobutton(root, text="Use Video File", variable=self.input_mode, value="video").pack(anchor="w", padx=20)

        # Camera dropdown (filled dynamically)
        self.camera_dropdown = ttk.Combobox(root, textvariable=self.selected_camera, state="readonly")
        self.update_camera_list()
        self.camera_dropdown.pack(padx=10, pady=5, fill="x")

        # Video file path (enabled only if 'video' is selected)
        ttk.Label(root, text="Video File Path:").pack(anchor="w", padx=10)
        self.path_entry = ttk.Entry(root, textvariable=self.video_path, state="disabled")
        self.path_entry.pack(padx=10, pady=5, fill="x")
        self.browse_button = ttk.Button(root, text="Browse", command=self.browse_file, state="disabled")
        self.browse_button.pack(pady=5)

        # Dropdown for FPS limits
        ttk.Label(root, text="FPS Limit:").pack(anchor="w", padx=10)
        self.fps_dropdown = ttk.Combobox(root, textvariable=self.fps_limit, values=["15", "30", "60"], state="readonly")
        self.fps_dropdown.pack(padx=10, pady=5, fill="x")

        # Start button
        ttk.Button(root, text="Start Detection", command=self.start_detection).pack(pady=20)

        # Binding input mode changes
        self.input_mode.trace("w", self.toggle_input_mode)

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

    def start_detection(self):
        """Start object detection with the selected options."""
        selected_model = self.selected_model.get()
        fps = int(self.fps_limit.get())

        log_file = "detections_log.json"  # JSON log file name

        if self.input_mode.get() == "camera":
            camera_index = self.selected_camera.get()
            process_video_or_camera(camera_index, YOLO(selected_model), fps_limit=fps, log_file=log_file)
        elif self.input_mode.get() == "video":
            video_path = self.video_path.get()
            if os.path.isfile(video_path):
                process_video_or_camera(video_path, YOLO(selected_model), fps_limit=fps, log_file=log_file)
            else:
                print("Invalid video file path")


def main():
    parser = argparse.ArgumentParser(description="YOLOv10 Object Detection")
    parser.add_argument("--no_gui", action="store_true", help="Run YOLO without GUI and directly via command-line arguments")
    parser.add_argument("--model", type=str, default="yolov10x.pt", help="Path to YOLO model file")
    parser.add_argument("--video_source", type=str, default="0", help="Path to the video file or camera index")
    parser.add_argument("--fps", type=int, default=30, help="FPS limit for the detection")
    parser.add_argument("--log_file", type=str, default="detections_log.json", help="File to save detections log")

    args = parser.parse_args()

    if args.no_gui:
        process_video_or_camera(args.video_source, YOLO(args.model), fps_limit=args.fps, log_file=args.log_file)
    else:
        root = tk.Tk()
        app = App(root)
        root.mainloop()


if __name__ == "__main__":
    main()
