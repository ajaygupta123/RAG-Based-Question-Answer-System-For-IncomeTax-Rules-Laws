import fitz # pymupdf
# import nltk
from transformers import BertTokenizer
import torch

# # Download NLTK sentence tokenizer
# nltk.download('punkt')


# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def read_text_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text


# changing path to drive
import os
os.chdir('/content/drive/MyDrive/colab/RAG/income_tax_sections_data/')

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
    # chunked_documents = [chunker.chunk_text(doc) for doc in documents]
    chunked_documents = chunker.chunk_text(documents)
    print("len chunked_documents", len(chunked_documents))
    # flat_chunked_documents = [chunk for sublist in chunked_documents for chunk in sublist]
    # print("len flat_chunked_documents", len(flat_chunked_documents))

    # # Text cleaning
    # text_cleaner = TextCleaner(remove_stopwords=False, use_stemming=False)
    # for i in range(len(flat_chunked_documents)):
    #   cleaned_text = text_cleaner.clean_text(flat_chunked_documents[i])
    #   flat_chunked_documents[i] = cleaned_text

    # Initialize retrieval and indexing systems
    bert_retrieval = BERTRetrieval() #DPRRetrieval() #BERTRetrieval()
    faiss_indexing = FaissIndexing(dimension=768) #AnnoyIndexing(dimension=100)

    # Initialize unified retrieval system
    retrieval_system = UnifiedRetrievalSystem(retrieval_system=bert_retrieval, indexing_system=faiss_indexing)

    # Index the documents
    retrieval_system.index_documents(chunked_documents) #(flat_chunked_documents)
    retrieval_system.save_index('faiss_index_2', 'documents.txt')

    # Load existing index and add new documents
    retrieval_system.load_index('faiss_index_2', 'documents.txt')
    # new_documents = ["New Document 1 text...", "New Document 2 text..."]
    # retrieval_system.update_index(new_documents)
    # retrieval_system.save_index('faiss_index', 'documents.txt')

    # Retrieval example with updated index
    query = "What is section 80GG?"
    top_documents = retrieval_system.retrieve(query)
    print("Updated Top Documents:", top_documents)


    # Generate response using DistilGPT-2
    generation_model = DistilGPT2Generation()
    context = " ".join(top_documents)
    answer = generation_model.generate(context, query)
    print("Generated Answer:", answer)