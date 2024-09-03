# embedding_module.py
import openai

# Set up your OpenAI API key
openai.api_key = ''

def get_embedding(formatted_string):
    """Get embeddings from OpenAI for the formatted OCR string."""
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=formatted_string
    )
    return response['data'][0]['embedding']
