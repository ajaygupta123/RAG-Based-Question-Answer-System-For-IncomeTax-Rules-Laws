import nltk
# Download NLTK sentence tokenizer
nltk.download('punkt')
from transformers import BertTokenizer

# Chunking strategy
class ChunkingStrategy:
    def __init__(self, model_name='bert-base-uncased', max_chunk_length=200, overlap=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_chunk_length = max_chunk_length
        self.overlap = overlap

    def chunk_text(self, text):
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(self.tokenizer.tokenize(sentence))
            if current_length + sentence_length > self.max_chunk_length:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-self.overlap:] if self.overlap else []
                current_length = sum(len(self.tokenizer.tokenize(sent)) for sent in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

# # Chunk the extracted text
# chunker = ChunkingStrategy()
# chunked_documents = [chunker.chunk_text(doc) for doc in documents]
# print("len chunked_documents", len(chunked_documents))
# flat_chunked_documents = [chunk for sublist in chunked_documents for chunk in sublist]
# print("len flat_chunked_documents", len(flat_chunked_documents))