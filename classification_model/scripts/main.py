
import os
from classification_module import process_directory_structure, save_embeddings, test, classify


def process_document(image_path, model_path='support_set.csv'):
    """Process and classify a document, then inspect it visually."""
    # Step 1: Classify the document
    test_embedding = test(image_path)
    classification = classify(test_embedding, model_path)
    print(f'Classified as: {classification[0]} with confidence {classification[1]:.4f}')

    # If you have the visual inspection function, call it here
    # inspection_results = inspect_with_yolo(image_path)
    # print('Visual inspection result:', inspection_results)

    return classification


def main():
    # Train on your dataset
    train_dir = '/home/amal/Documents/gen_ai/v5/dirs/train'
    labels, embeddings = process_directory_structure(train_dir)
    save_embeddings(labels, embeddings)

    # Test with a new image
    test_image_path = '/home/amal/Documents/gen_ai/v5/dirs/main_output_folder/CERTFICATE BY COMPANY SECRETARY REGARDING BORROWING_LIMIT OF THE BOARD OF DIRECTORS/CERTFICATE BY COMPANY SECRETARY REGARDING BORROWING_LIMIT OF THE BOARD OF DIRECTORS_1.jpg'
    process_document(test_image_path)


if __name__ == '__main__':
    main()
