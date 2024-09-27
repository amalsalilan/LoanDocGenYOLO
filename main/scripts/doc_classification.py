import os
import pandas as pd
import numpy as np
import fitz  # PyMuPDF for PDF handling
import pytesseract
import openai
from numpy.linalg import norm
import shutil  # For copying files instead of moving

# Set up your OpenAI API key
openai.api_key = ''

# Configure Tesseract for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:/Users/aabid/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'

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

# Function to extract text from PDF using PyMuPDF
def pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text("text")  # Extract text in a simple format
    
    doc.close()
    return full_text

# Function to apply OCR and get embedding for a test PDF
def test(pdf_path):
    # Extract text from PDF using PyMuPDF
    extracted_text = pdf_to_text(pdf_path)
    
    # In case the PDF contains images or has embedded text that requires OCR
    if not extracted_text.strip():
        print(f"No text detected in {pdf_path}, consider using OCR.")
        # Add fallback OCR process if necessary (optional)
    
    # Get embedding of the extracted text
    embedding = get_embedding(extracted_text)
    
    return embedding

# Function to retrieve all document labels in the support set CSV
def get_support_documents(model_path='support_set.csv'):
    model = pd.read_csv(model_path)
    document_list = model['Label'].tolist()  # Assuming the CSV has a 'Label' column for document names/labels
    return document_list

# Function to get unique support documents, excluding 'detection'
def get_unique_support_documents(model_path='support_set.csv'):
    document_list = get_support_documents(model_path)
    filtered_documents = [doc for doc in document_list if not doc.startswith('detection')]
    unique_documents = list(set(filtered_documents))
    return unique_documents

# Main function to classify all PDFs in a folder and store predicted documents
def main():
    confidence_threshold = 0.85  # Set your confidence threshold here

    # Get the current script directory and set the main directory as the root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.abspath(os.path.join(script_dir, '..'))  # Move one level up to reach 'main'

    # Set the folder paths relative to the main directory
    input_folder = os.path.join(main_dir, 'dirs', 'client_documents')  # Input folder
    verified_output_folder = os.path.join(main_dir, 'dirs', 'classified_documents')  # Verified documents output folder
    unverified_output_folder = os.path.join(main_dir, 'dirs', 'unclassified_documents')  # Unverified documents output folder

    # Ensure output directories exist
    os.makedirs(verified_output_folder, exist_ok=True)
    os.makedirs(unverified_output_folder, exist_ok=True)

    # Get the list of unique documents in the support set
    unique_support_documents = get_unique_support_documents()

    # Variable to store the predicted document classes (only those above confidence threshold)
    predicted_documents = []

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

            # Only add the document to the predicted_documents if the confidence is above the threshold
            if confidence > confidence_threshold:
                print(f'Classified as: {detected_label} with confidence {confidence:.4f}')
                
                # Copy the PDF to the verified documents folder with the detected label as the filename
                output_pdf_path = os.path.join(verified_output_folder, f'{detected_label}.pdf')
                shutil.copy(test_pdf_path, output_pdf_path)
                print(f'Copied the PDF to {output_pdf_path}')
                
                # Add the predicted document to the list
                predicted_documents.append(detected_label)
            else:
                print(f'Confidence {confidence:.4f} is too low, marking document as unverified.')
                
                # Copy the PDF to the unverified documents folder without renaming
                shutil.copy(test_pdf_path, unverified_output_folder)
                print(f'Copied the document to {unverified_output_folder}')

   
    # Compare the predicted documents with the unique support documents
    missing_documents = set(unique_support_documents) - set(predicted_documents)
    print("Missing documents:", missing_documents)

    # Save missing documents to a text file in the classified documents folder
    missing_docs_file = os.path.join(verified_output_folder, 'missing_documents.txt')
    with open(missing_docs_file, 'w') as f:
        for doc in missing_documents:
            f.write(doc + '\n')

    print(f'Missing documents saved to {missing_docs_file}')

if __name__ == '__main__':
    main()
