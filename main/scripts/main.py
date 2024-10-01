import subprocess

def run_script(script_name):
    try:
        # Run the Python script using subprocess
        result = subprocess.run(['python3', script_name], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f'Error occurred while running {script_name}: {e}')

def main():
    # Step 1: Run doc_classification.py
    run_script('doc_classification.py')

    run_script('missing_doc.py')

    # Step 2: Run document_to_img.py
    run_script('document_to_img.py')

if __name__ == '__main__':
    main()
