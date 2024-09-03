# classification_module.py
import os
import pandas as pd
import numpy as np
from numpy.linalg import norm

from ocr_module import apply_ocr, text_to_string_encode
from embedding_module import get_embedding

def process_directory_structure(train_dir):
    """Process a directory of training images and generate embeddings."""
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

def save_embeddings(labels, embeddings, output_path='support_set.csv'):
    """Save the generated embeddings and labels to a CSV file."""
    df = pd.DataFrame({'Label': labels, 'Embedding': embeddings})
    df.to_csv(output_path, index=False)

def classify(test_embedding, model_path='support_set.csv'):
    """Classify a test embedding against the saved embeddings."""
    model = pd.read_csv(model_path)
    similarities = []

    for idx, row in model.iterrows():
        train_embedding = np.array(eval(row['Embedding']))
        similarity = np.dot(train_embedding, test_embedding) / (norm(train_embedding) * norm(test_embedding))
        similarities.append((row['Label'], similarity))

    return max(similarities, key=lambda x: x[1])

def test(image_path):
    """Test a single image by generating its embedding and returning it."""
    ocr_data = apply_ocr(image_path)
    min_x, min_y = ocr_data['left'].min(), ocr_data['top'].min()
    max_x, max_y = (ocr_data['left'] + ocr_data['width']).max(), (ocr_data['top'] + ocr_data['height']).max()
    formatted_string = text_to_string_encode(
        zip(ocr_data['left'], ocr_data['top'], ocr_data['text']),
        min_x, min_y, max_x, max_y
    )
    return get_embedding(formatted_string)
