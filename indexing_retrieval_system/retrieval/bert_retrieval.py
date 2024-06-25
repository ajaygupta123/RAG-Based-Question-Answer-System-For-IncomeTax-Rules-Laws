from transformers import BertTokenizer, BertModel, DPRContextEncoder, DPRQuestionEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer
from retrieval_interface import RetrievalInterface
from utils.chunking_strategy import  ChunkingStrategy
import torch


class BERTRetrieval(RetrievalInterface):
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.chunker = ChunkingStrategy(model_name=model_name)

    # def encode_documents(self, documents):
    #     all_embeddings = []
    #     for doc in documents:
    #         chunks = self.chunker.chunk_text(doc)
    #         for chunk in chunks:
    #             inputs = self.tokenizer(chunk, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    #             with torch.no_grad():
    #                 outputs = self.model(**inputs)
    #             all_embeddings.append(outputs.last_hidden_state.mean(dim=1))
    #     return torch.cat(all_embeddings, dim=0)

    def encode_documents(self, documents):
        embeddings = []
        for doc in documents:
            inputs = self.tokenizer(doc, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1))
        return torch.cat(embeddings, dim=0)

    def encode_query(self, query):
        inputs = self.tokenizer(query, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)
