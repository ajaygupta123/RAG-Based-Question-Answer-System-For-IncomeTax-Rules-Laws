from transformers import BertTokenizer, BertModel, DPRContextEncoder, DPRQuestionEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer
from retrieval_interface import RetrievalInterface
from utils.chunking_strategy import  ChunkingStrategy
import torch

class DPRRetrieval(RetrievalInterface):
    def __init__(self):
        self.context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        self.question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.chunker = ChunkingStrategy(model_name='facebook/dpr-ctx_encoder-single-nq-base')

    # def encode_documents(self, documents):
    #     all_embeddings = []
    #     for doc in documents:
    #         chunks = self.chunker.chunk_text(doc)
    #         inputs = self.context_tokenizer(chunks, padding=True, truncation=True, return_tensors='pt')
    #         with torch.no_grad():
    #             embeddings = self.context_encoder(**inputs).pooler_output
    #         all_embeddings.append(embeddings)
    #     return torch.cat(all_embeddings, dim=0)

    def encode_documents(self, documents):
        inputs = self.context_tokenizer(documents, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            embeddings = self.context_encoder(**inputs).pooler_output
        return embeddings

    def encode_query(self, query):
        inputs = self.question_tokenizer(query, return_tensors='pt')
        with torch.no_grad():
            embedding = self.question_encoder(**inputs).pooler_output
        return embedding
