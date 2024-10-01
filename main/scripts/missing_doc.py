import os
import fitz  # PyMuPDF

# Function to extract the first heading or first line if no heading is found
def extract_primary_heading(pdf_path):
    pdf_document = fitz.open(pdf_path)
    headings = []
    first_non_empty_line = None  # Fallback to store the first line if no heading is found

    # Iterate over each page in the PDF
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)  # Load the page
        text = page.get_text("text")  # Extract text
        first_non_numeric_heading = None
        
        # Iterate through each line of text on the page
        for line in text.split('\n'):
            stripped_line = line.strip()

            # Save the first non-empty line as a fallback
            if not first_non_empty_line and stripped_line:
                first_non_empty_line = stripped_line

            # Skip empty lines and purely numeric lines, find heading
            if stripped_line.isupper() and stripped_line and not stripped_line.isdigit():
                if len(stripped_line) > 3:  # Avoid very short words like 'AND', 'OF'
                    first_non_numeric_heading = stripped_line
                    break
        
        # If a heading is found, append and break
        if first_non_numeric_heading:
            headings.append(first_non_numeric_heading)
            break  # Stop after extracting the first meaningful heading

    pdf_document.close()

    # If no heading is found, fallback to the first non-empty line
    if not headings and first_non_empty_line:
        headings.append(first_non_empty_line)
    
    return headings

# Function to read the missing documents from a file
def read_missing_documents(missing_docs_file):
    with open(missing_docs_file, 'r') as f:
        missing_documents = [line.strip() for line in f.readlines()]
    return missing_documents

# Function to extract headings from unclassified documents and save final message to a text file
def save_uploaded_headings_to_file(unclassified_folder, missing_documents, output_file):
    with open(output_file, 'w') as f:
        for idx, file_name in enumerate(os.listdir(unclassified_folder)):
            if file_name.endswith('.pdf'):
                pdf_path = os.path.join(unclassified_folder, file_name)
                headings = extract_primary_heading(pdf_path)

                # If a heading is found, write the message with extracted heading and missing document name
                if idx < len(missing_documents):
                    missing_doc = missing_documents[idx]
                    extracted_heading = headings[0] if headings else 'Unknown Heading'
                    # Concatenate the file name with the heading
                    output_text = f"You have uploaded '{file_name}: {extracted_heading}' instead of '{missing_doc}'.\n"
                    f.write(output_text)

# Main function to execute the process
def main():
    # Get the current script directory and set the main directory as the root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.abspath(os.path.join(script_dir, '..'))  # Move one level up to reach 'main'

    # Directory containing unclassified PDFs
    unclassified_folder = os.path.join(main_dir, 'dirs', 'unclassified_documents')
    
    # Path to missing documents file
    missing_docs_file = os.path.join(main_dir, 'dirs', 'classified_documents', 'missing_documents.txt')
    
    # Path to the output text file
    output_file = os.path.join(main_dir, 'dirs', 'classified_documents', 'uploaded_headings.txt')
    
    # Read missing documents
    missing_documents = read_missing_documents(missing_docs_file)
    
    # Save the final output to a text file
    save_uploaded_headings_to_file(unclassified_folder, missing_documents, output_file)

if __name__ == "__main__":
    main()
