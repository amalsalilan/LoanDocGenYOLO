import os
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import openai
from numpy.linalg import norm

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

# Function to apply OCR and get embedding for a test image
def test(image_path):
    ocr_data = apply_ocr(image_path)
    min_x, min_y = ocr_data['left'].min(), ocr_data['top'].min()
    max_x, max_y = (ocr_data['left'] + ocr_data['width']).max(), (ocr_data['top'] + ocr_data['height']).max()
    formatted_string = text_to_string_encode(
        zip(ocr_data['left'], ocr_data['top'], ocr_data['text']),
        min_x, min_y, max_x, max_y
    )
    return get_embedding(formatted_string)

# Function to classify a test embedding
def classify(test_embedding, model_path='support_set.csv'):
    model = pd.read_csv(model_path)
    similarities = []

    for idx, row in model.iterrows():
        train_embedding = np.array(eval(row['Embedding']))
        similarity = np.dot(train_embedding, test_embedding) / (norm(train_embedding) * norm(test_embedding))
        similarities.append((row['Label'], similarity))

    return max(similarities, key=lambda x: x[1])

# Main function to process training data and classify a test image
def main():
    train_dir = '/home/amal/Documents/gen_ai/v5/dirs/train'  # Specify your train directory path here
    labels, embeddings = process_directory_structure(train_dir)
    save_embeddings(labels, embeddings)

    test_image_path = '/home/amal/Documents/gen_ai/v5/dirs/main_output_folder/CERTFICATE BY COMPANY SECRETARY REGARDING BORROWING_LIMIT OF THE BOARD OF DIRECTORS/CERTFICATE BY COMPANY SECRETARY REGARDING BORROWING_LIMIT OF THE BOARD OF DIRECTORS_1.jpg'  # Specify your test image path here
    test_embedding = test(test_image_path)
    classification = classify(test_embedding)
    print(f'Classified as: {classification[0]} with confidence {classification[1]:.4f}')

if __name__ == '__main__':
    main()
