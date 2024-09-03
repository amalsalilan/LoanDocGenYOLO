# ocr_module.py
import pandas as pd
from PIL import Image
import pytesseract

# Configure Tesseract for Ubuntu
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def apply_ocr(image_path):
    """Apply OCR to an image and return a dataframe."""
    image = Image.open(image_path)
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
    return ocr_data

def text_to_string_encode(coordinates, min_x, min_y, max_x, max_y):
    """Encode OCR text into a formatted string."""
    matrix = [[0] * 50 for _ in range(50)]
    for x, y, z in coordinates:
        if pd.notna(z):
            matrix_x = int((x - min_x) / (max_x - min_x) * (50 - 1)) if max_x != min_x else 0
            matrix_y = int((y - min_y) / (max_y - min_y) * (50 - 1)) if max_y != min_y else 0
            matrix[matrix_y][matrix_x] = z
    return '/'.join([''.join(map(str, line)) for line in matrix])
