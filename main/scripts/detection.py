from collections import Counter
from ultralytics import YOLO
import os
import shutil

# Get the current script directory and set the main directory as the root
script_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.abspath(os.path.join(script_dir, '..'))  # Move one level up to reach 'main'

# Main folder paths relative to the main directory
main_folder = os.path.join(main_dir, 'dirs', 'pdf_to_image_conversion')
output_folder = os.path.join(main_dir, 'dirs', 'detection_results')
completed_folder = os.path.join(main_dir, 'dirs', 'verified_documents')
verified_folder = os.path.join(main_dir, 'dirs', 'classified_documents')  # Verified documents folder

# Load the YOLO model
model = YOLO(os.path.join(main_dir, 'models', 'train5', 'weights', 'best.pt'))

# Class to exclude from detection (e.g., 'signature')
class_to_exclude = 'signature'

# Iterate through the main folder
for root, dirs, files in os.walk(main_folder):
    for dir in dirs:
        # Construct the source directory for each subfolder
        source_dir = os.path.join(root, dir)

        # Perform object detection on the source directory
        results = model(source=source_dir, conf=0.65)

        # Flag to check if there are any detections in this folder
        detections_found = False

        # Counter for saving images with detections
        counter = 1

        # Prepare to collect detection information
        detection_summary = []

        # Iterate through results
        for result in results:
            # Get class names of detected objects, excluding the unwanted class
            detected_classes = [model.names[int(cls)] for cls in result.boxes.cls if model.names[int(cls)] != class_to_exclude]

            # If there are no detections (after filtering out 'signature'), continue to the next result
            if not detected_classes:
                continue

            # If we find relevant detections, mark that detections were found
            if detected_classes:
                detections_found = True
                output_subfolder = os.path.join(output_folder, f"detection_{dir}")
                os.makedirs(output_subfolder, exist_ok=True)
                
                # Save the detected images to the output folder
                image_file_name = f"{dir}_page{counter}.jpg"
                result.save(filename=f"{output_subfolder}/{image_file_name}")
                
                # Count occurrences of each detected class
                class_counts = Counter(detected_classes)  # Use Counter to get counts
                
                # Prepare the summary for the current image
                class_summary = ', '.join([f"{count} {cls_name}" for cls_name, count in class_counts.items()])
                detection_summary.append(f"Image: {image_file_name} - Detected: {class_summary}")
                
                counter += 1

        # If no detections were found for any images in the folder, move the corresponding document
        if not detections_found:
            print(f"No detections in folder '{dir}'. Moving the associated document from verified folder to completed folder.")

            # Look for the associated document in the verified_folder
            doc_file = None
            for file in os.listdir(verified_folder):
                # Assuming the document name corresponds to the folder name, check for matching names with .pdf or .docx extensions
                if file.startswith(dir) and file.endswith(('.pdf', '.docx')):
                    doc_file = os.path.join(verified_folder, file)
                    break

            if doc_file:
                # Move the document file to the completed folder
                shutil.move(doc_file, os.path.join(completed_folder, os.path.basename(doc_file)))
                print(f"Document '{doc_file}' moved to completed folder.")
            else:
                print(f"No matching document found in the verified folder for '{dir}'.")

        else:
            print(f"Detections found in folder '{dir}'. Saved to detection folder.")
            
            # If there are detections, create a summary text file
            summary_file_path = os.path.join(output_subfolder, f"detection_summary_{dir}.txt")
            with open(summary_file_path, 'w') as summary_file:
                summary_file.write("\n".join(detection_summary))
            print(f"Detection summary saved to {summary_file_path}.")

# Final completion message
print("Detection process completed.")
