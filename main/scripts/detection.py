
from ultralytics import YOLO
import os
import shutil
current_dir=os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Main folder
main_folder ='/home/amal/Documents/gen_ai/v5/dirs/main_output_folder'
# folder name
output_folder = '/home/amal/Documents/gen_ai/v5/dirs/detections'
completed_folder='/home/amal/Documents/gen_ai/v5/dirs/processed_documents'
# load the model
model = YOLO('/home/amal/Documents/gen_ai/v5/models/train5/weights/best.pt')


for root, dirs, files in os.walk(main_folder):
    for dir in dirs:
        # Construct the source directory for each subfolder
        source_dir = os.path.join(root, dir)

        # Get the total number of images in the folder
        total_images = len([f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))])

        # Perform object detection on the source directory
        results = model(source=source_dir)

        # Check if any images were detected
        if len(results) > 0:
            print(f"Images in folder '{dir}' were detected:")
            # Create a new directory to save output images with the same input folder name
            output_subfolder = os.path.join(output_folder, f"detection_{dir}")
            os.makedirs(output_subfolder, exist_ok=True)

            # Save the output images with consecutive page numbers
            counter = 1
            for result in results:
                result.save(filename=f"{output_subfolder}/{dir}_page{counter}.jpg")
                counter += 1
        else:
            print(f"No images in folder '{dir}' were detected.")
            # Move the folder to the completed documents directory if no images were detected
            shutil.move(source_dir, os.path.join(completed_folder, dir))
            os.makedirs(completed_folder, exist_ok=True)
print("Detection Completed")























