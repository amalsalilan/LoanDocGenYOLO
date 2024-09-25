import os
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import openai
from pdf2image import convert_from_path
from numpy.linalg import norm
import shutil  # For copying files instead of moving

# Set up your OpenAI API key
openai.api_key = ''

# Configure Tesseract for Ubuntu
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Function to apply OCR
def apply_ocr(image):
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
    return ocr_data

# Function to encode OCR text into a formatted string
def text_to_string_encode(coordinates, min_x, min_y, max_x, max_y):
    matrix = [[0] * 50 for _ in range(50)]
    for x, y, z in coordinates:
        if pd.notna(z):
            matrix_x = int((x - min_x) / (max_x - min_x) * (50 - 1)) if max_x != min_x else 0
            matrix_y = int((y - min_y) / (max_y - min_y) * (50 - 1)) if max_y != min_y else 0
            matrix[matrix_y][matrix_x] = z
    return '/'.join([''.join(map(str, line)) for line in matrix])

# Function to get embeddings from OpenAI
def get_embedding(formatted_string):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=formatted_string
    )
    return response['data'][0]['embedding']

# Function to classify a test embedding
def classify(test_embedding, model_path='support_set.csv'):
    model = pd.read_csv(model_path)
    similarities = []

    for idx, row in model.iterrows():
        train_embedding = np.array(eval(row['Embedding']))
        similarity = np.dot(train_embedding, test_embedding) / (norm(train_embedding) * norm(test_embedding))
        similarities.append((row['Label'], similarity))

    return max(similarities, key=lambda x: x[1])

# Function to convert PDF to images
def pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)

# Function to apply OCR and get embedding for a test PDF
def test(pdf_path):
    images = pdf_to_images(pdf_path)
    all_text_embeddings = []
    
    for image in images:
        ocr_data = apply_ocr(image)
        min_x, min_y = ocr_data['left'].min(), ocr_data['top'].min()
        max_x, max_y = (ocr_data['left'] + ocr_data['width']).max(), (ocr_data['top'] + ocr_data['height']).max()
        
        formatted_string = text_to_string_encode(
            zip(ocr_data['left'], ocr_data['top'], ocr_data['text']),
            min_x, min_y, max_x, max_y
        )
        embedding = get_embedding(formatted_string)
        all_text_embeddings.append(embedding)
    
    # Return the first embedding for simplicity, assuming one document per PDF
    return all_text_embeddings[0]

# Main function to classify all PDFs in a folder
def main():
    confidence_threshold = 0.4  # Set your confidence threshold here
    
    input_folder = '/home/amal/Documents/gen_ai/main/dirs/client/'  # Specify your input folder here
    verified_output_folder = '/home/amal/Documents/gen_ai/main/dirs/verified_documents/'  # Verified documents output folder
    unverified_output_folder = '/home/amal/Documents/gen_ai/main/dirs/unverified_documents/'  # Unnecessary documents output folder

    # Ensure output directories exist
    os.makedirs(verified_output_folder, exist_ok=True)
    os.makedirs(unverified_output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.pdf'):  # Only process PDFs
            test_pdf_path = os.path.join(input_folder, file_name)
            print(f'Processing: {test_pdf_path}')
            
            # Get the embedding and classify the document
            test_embedding = test(test_pdf_path)
            classification = classify(test_embedding)

            detected_label = classification[0]
            confidence = classification[1]

            if confidence > confidence_threshold:
                print(f'Classified as: {detected_label} with confidence {confidence:.4f}')
                
                # Copy the PDF to the verified documents folder with the detected label as the filename
                output_pdf_path = os.path.join(verified_output_folder, f'{detected_label}.pdf')
                shutil.copy(test_pdf_path, output_pdf_path)
                print(f'Copied the PDF to {output_pdf_path}')
            else:
                print(f'Confidence {confidence:.4f} is too low, marking document as unverified.')
                
                # Copy the PDF to the unverified documents folder without renaming
                shutil.copy(test_pdf_path, unverified_output_folder)
                print(f'Copied the document to {unverified_output_folder}')

if __name__ == '__main__':
    main()
