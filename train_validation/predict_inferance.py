import matplotlib.pyplot as plt
from ultralytics import YOLO
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def predict_on_image(model, image_path):
    """
    Run prediction on a single image
    """
    # Run prediction
    results = model(image_path)

    # Plot results
    result_img = results[0].plot()

    # Display results
    plt.figure(figsize=(10, 8))
    plt.imshow(result_img)
    plt.axis('off')
    plt.show()


model_path= 'weights/best_small_object_detector.pt'
yolo_model = YOLO(model_path)
# Example usage (uncomment to use):
predict_on_image(yolo_model, "Dataset/val/images/Subject_58/280.png")
