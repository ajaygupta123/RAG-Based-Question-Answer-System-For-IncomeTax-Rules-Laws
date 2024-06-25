# test the chunking and text cleaning part

import fitz # pymupdf
# import nltk
from transformers import BertTokenizer
import torch



# changing path to drive
import os
os.chdir('/content/drive/MyDrive/colab/RAG/income_tax_sections_data/')


def read_text_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text



# Example usage
if __name__ == "__main__":
    # # Extract text from PDFs
    # pdf_paths = ['section 80G.pdf', 'section 80GG.pdf']
    # documents = [extract_text_from_pdf(path) for path in pdf_paths]
    # print("len documents", len(documents))
    # # print(documents[0])

    # Extract text from .txt file
    txt_file_paths = '1000_answers.txt'
    documents = read_text_file(txt_file_paths)
    print("len documents", len(documents))
    # print(documents[0])

    # Chunk the extracted text
    chunker = ChunkingStrategy()
    chunked_documents = chunker.chunk_text(documents) #[chunker.chunk_text(doc) for doc in documents]
    print("len chunked_documents", len(chunked_documents))
    flat_chunked_documents = [chunk for sublist in chunked_documents for chunk in sublist]
    print("len flat_chunked_documents", len(flat_chunked_documents))
