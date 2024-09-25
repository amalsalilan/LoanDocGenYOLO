import os
import fitz  # PyMuPDF
from PIL import Image
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.abspath(os.path.join(script_dir, '..')) 

def process_pdfs(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Create the main output directory

    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)

            document_name = filename.split('.')[0]  # Get document name without extension
            document_output_dir = os.path.join(output_dir, document_name)
            os.makedirs(document_output_dir, exist_ok=True)  # Create folder for this document

            doc = fitz.open(pdf_path)

            for page_index in range(len(doc)):
                page = doc.load_page(page_index)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                image_filename = f"{document_name}_{page_index+1}.jpg"  # Image name with page number
                image_path = os.path.join(document_output_dir, image_filename)
                img.save(image_path)

if __name__ == "__main__":
    input_directory = os.path.join(main_dir, 'dirs', 'classified_documents')
    output_directory = os.path.join(main_dir, 'dirs', 'pdf_to_image_conversion')
    process_pdfs(input_directory, output_directory)



def main():
    print("PDF To Image Converter Completed")
    print("Detection Started")
    command = ["python", "detection.py"]
    subprocess.run(command)

if __name__ == "__main__":
    main()


