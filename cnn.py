import torch
import torchvision
from torchvision import transforms
import cv2
import time
import argparse
import json
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

# Load the pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()  # Set to evaluation mode

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define transformation to normalize the image for Faster R-CNN
transform = transforms.Compose([transforms.ToTensor()])

# Function to log detections and inference time
def log_detections(log_file, detections):
    """Log the detection results and inference times to a JSON file."""
    with open(log_file, 'w') as f:
        json.dump(detections, f, indent=4)
    print(f"Detections logged to {log_file}")

def process_image(image, threshold=0.5):
    """
    Process the image using Faster R-CNN and return the detections.
    :param image: The input image in BGR format.
    :param threshold: Confidence threshold to filter weak detections.
    """
    # Convert BGR (OpenCV format) to RGB and apply transformations
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image_rgb).to(device).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        # Run Faster R-CNN inference
        start_time = time.time()
        outputs = model(image_tensor)
        end_time = time.time()

    # Extract the bounding boxes and scores
    outputs = outputs[0]  # Batch size is 1, so we extract the first output
    boxes = outputs['boxes'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()

    # Filter out detections with confidence scores below the threshold
    filtered_boxes = boxes[scores >= threshold]
    filtered_labels = labels[scores >= threshold]
    filtered_scores = scores[scores >= threshold]

    # Inference time
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.3f} seconds")

    return filtered_boxes, filtered_labels, filtered_scores, inference_time

def visualize_detections(image, boxes, labels, scores):
    """
    Draw bounding boxes and labels on the image.
    :param image: The input image in BGR format.
    :param boxes: Bounding boxes from the Faster R-CNN output.
    :param labels: Class labels from the Faster R-CNN output.
    :param scores: Confidence scores from the Faster R-CNN output.
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
        # Check if label index is within the valid range
        if labels[i] >= len(COCO_INSTANCE_CATEGORY_NAMES):
            print(f"Warning: Label index {labels[i]} is out of range.")
            continue  # Skip this detection
        label = COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
        score = scores[i]
        x1, y1, x2, y2 = map(int, box)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put label and confidence score
        label_text = f"{label}: {score:.2f}"
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def run_video_detection(video_source=0, threshold=0.5, log_file="faster_rcnn_log.json"):
    """
    Run Faster R-CNN detection on a video source (webcam or video file).
    Log results per frame.
    """
    if isinstance(video_source, str):
        video_source = video_source.replace('\\', '/')  # Correct path for Windows

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    print("Video source opened successfully!")

    detections = []  # To store log information

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read a frame or end of video.")
            break

        # Get current frame number for logging
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Run inference on the current frame
        boxes, labels, scores, inference_time = process_image(frame, threshold)

        # Save detection information
        detections.append({
            "frame_number": frame_number,  # Log frame number
            "inference_time": inference_time,
            "boxes": boxes.tolist(),
            "labels": labels.tolist(),
            "scores": scores.tolist()
        })

        # Visualize the detections
        output_frame = visualize_detections(frame.copy(), boxes, labels, scores)

        # Display the output
        cv2.imshow("Faster R-CNN Detection", output_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Log detections
    log_detections(log_file, detections)

if __name__ == "__main__":
    # Argument parser to pass video source as a command-line argument
    parser = argparse.ArgumentParser(description="Faster R-CNN Object Detection")
    parser.add_argument("--video_source", type=str, default="0", help="Path to the video file or camera index (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--log_file", type=str, default="faster_rcnn_log.json", help="File to log detections and inference times")
    
    args = parser.parse_args()

    # Test the Faster R-CNN model on video
    run_video_detection(video_source=args.video_source, threshold=args.threshold, log_file=args.log_file)
