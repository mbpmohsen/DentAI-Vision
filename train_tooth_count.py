from ultralytics import YOLO

# Step 1: Train the Model
def train_model():
    """
    Train a YOLOv8 model using the provided dataset.
    """
    # Load the YOLOv8 model (pre-trained weights)
    model = YOLO('yolov8n.pt')  # 'yolov8n.pt' is the nano model (small and fast)

    # Train the model
    model.train(
        data='./datasets/data.yaml',  # Path to your data.yml file
        epochs=50,                # Number of training epochs
        imgsz=640,                # Image size
        batch=8                   # Batch size
    )
    print("Training completed. Model is saved in the 'runs/' directory.")


# Step 2: Test the Model
def test_model(image_path, model_path='runs/detect/train/weights/best.pt'):
    """
    Test the trained YOLOv8 model on a specific image.

    Args:
        image_path (str): Path to the test image.
        model_path (str): Path to the trained model weights. Default is 'best.pt' after training.
    """
    # Load the trained model
    model = YOLO(model_path)

    # Perform inference
    results = model(image_path)

    # Display results
    results.show()

    # Count the number of 'tooth' detections (Class ID 0)
    detections = results[0].boxes  # Detected bounding boxes
    tooth_count = len([box for box in detections if box.cls == 0])  # Class ID 0 is 'tooth'

    print(f"Total teeth detected: {tooth_count}")
    return tooth_count


# Step 3: Main Script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and test a YOLOv8 model for tooth counting.")
    parser.add_argument('--train', action='store_true', help="Train the model.")
    parser.add_argument('--test', action='store_true', help="Test the model.")
    parser.add_argument('--image', type=str, default=None, help="Path to the image for testing.")
    parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.pt', help="Path to the trained model weights.")

    args = parser.parse_args()

    if args.train:
        train_model()
    if args.test:
        if args.image is None:
            print("Error: Please specify an image path using --image for testing.")
        else:
            test_model(args.image, args.model)
