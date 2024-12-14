import torch
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO  # Use the YOLO class from the ultralytics package

# Initialize FastAPI app
train_tooth_count_app = FastAPI()

# Function to load the model
def load_model(model_path='/Users/mohsen/Projects/my_tooth_counter/runs/detect/train10/weights/best.pt'):
    try:
        # Use the ultralytics YOLO class to load the model
        model = YOLO(model_path)  # Load the YOLO model from the path
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Prediction route for FastAPI
@train_tooth_count_app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        image = np.array(image)

        # Ensure image is in RGB format for YOLO model
        if image.ndim == 2:  # If grayscale, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Load the YOLO model
        model = load_model()

        if model:
            # Perform inference
            results = model(image)  # Perform inference using the YOLO model

            # The results are a list, but we need to access the 'xyxy' attribute of the first result
            result = results[0]  # Get the first (and typically only) result in the list

            # Get predictions (x1, y1, x2, y2, confidence, class_id) from bounding boxes
            predicted_boxes = result.boxes.xyxy.numpy()  # Bounding boxes

            # Draw bounding boxes on the image (for visualization)
            image_with_boxes = image.copy()
            for box in predicted_boxes:
                # If the box contains only 4 values, it likely doesn't have confidence or class_id
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    confidence = 1.0  # Default confidence value when not provided
                    class_id = 0  # Default class id when not provided
                else:
                    x1, y1, x2, y2, confidence, class_id = box

                # Draw bounding box with the default or actual values
                cv2.rectangle(image_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Convert the image with boxes back to byte format to send as response
            _, buffer = cv2.imencode('.png', image_with_boxes)
            byte_image = buffer.tobytes()

            # Return image with bounding boxes
            return StreamingResponse(BytesIO(byte_image), media_type="image/png", headers={"Content-Disposition": "inline; filename=image.png"})
        else:
            return {"error": "Model loading failed."}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
