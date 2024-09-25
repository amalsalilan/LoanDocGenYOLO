import os
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import openai

# Set up your OpenAI API key
openai.api_key = ''

# Configure Tesseract for Ubuntu
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Function to apply OCR
def apply_ocr(image_path):
    image = Image.open(image_path)
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

# Function to process the directory structure and generate embeddings
def process_directory_structure(train_dir):
    labels = []
    embeddings = []
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                class_name = os.path.basename(root)
                image_path = os.path.join(root, file)
                ocr_data = apply_ocr(image_path)
                min_x, min_y = ocr_data['left'].min(), ocr_data['top'].min()
                max_x, max_y = (ocr_data['left'] + ocr_data['width']).max(), (ocr_data['top'] + ocr_data['height']).max()
                formatted_string = text_to_string_encode(
                    zip(ocr_data['left'], ocr_data['top'], ocr_data['text']),
                    min_x, min_y, max_x, max_y
                )
                embedding = get_embedding(formatted_string)
                labels.append(class_name)
                embeddings.append(embedding)
    return labels, embeddings

# Function to save embeddings to CSV
def save_embeddings(labels, embeddings, output_path='support_set.csv'):
    df = pd.DataFrame({'Label': labels, 'Embedding': embeddings})
    df.to_csv(output_path, index=False)

# Main function to process training data and generate embeddings
def main():
    train_dir = '/home/amal/Documents/gen_ai/main/dirs/'  # Specify your train directory path here
    labels, embeddings = process_directory_structure(train_dir)
    save_embeddings(labels, embeddings)
    print("Training completed and embeddings saved.")

if __name__ == '__main__':
    main()
