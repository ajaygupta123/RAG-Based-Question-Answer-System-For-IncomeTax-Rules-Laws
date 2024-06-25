# import faiss
# import numpy as np
# import torch

# class FaissIndexing(IndexingInterface):
#     def __init__(self, dimension, use_gpu=True):
#         self.dimension = dimension
#         self.use_gpu = use_gpu
#         self.documents = []

#         if use_gpu:
#             res = faiss.StandardGpuResources()  # use a single GPU
#             self.index = faiss.GpuIndexFlatL2(res, dimension)  # build the index
#         else:
#             self.index = faiss.IndexFlatL2(dimension)  # build the index

#     def add_embeddings(self, embeddings, documents):
#         if self.use_gpu:
#             embeddings = embeddings.to('cuda')  # move embeddings to GPU if necessary
#         self.index.add(embeddings.cpu().numpy())
#         self.documents.extend(documents)

#     def save_index(self, index_path, doc_path):
#         faiss.write_index(faiss.index_cpu_to_all_gpus(self.index), index_path) if self.use_gpu else faiss.write_index(self.index, index_path)
#         with open(doc_path, 'w') as f:
#             for doc in self.documents:
#                 f.write(f"{doc}\n")

#     def load_index(self, index_path, doc_path):
#         self.index = faiss.read_index(index_path)
#         if self.use_gpu:
#             res = faiss.StandardGpuResources()
#             self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
#         with open(doc_path, 'r') as f:
#             self.documents = [line.strip() for line in f.readlines()]

#     def search(self, query_embedding, top_n=5):
#         if self.use_gpu:
#             query_embedding = query_embedding.to('cuda')  # move query to GPU if necessary
#         D, I = self.index.search(query_embedding.cpu().numpy().reshape(1, -1), top_n)
#         return [self.documents[i] for i in I[0]]

#     def update_index(self, documents):
#         embeddings = self.encode_documents(documents)
#         self.add_embeddings(embeddings, documents)



###cpu based code

import faiss
import numpy as np
from indexing_interface import IndexingInterface

class FaissIndexing(IndexingInterface):
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    def add_embeddings(self, embeddings, documents):
        self.index.add(embeddings.cpu().numpy())
        self.documents.extend(documents)

    def save_index(self, index_path, doc_path):
        faiss.write_index(self.index, index_path)
        with open(doc_path, 'w') as f:
            for doc in self.documents:
                f.write(f"{doc}\n")

    def load_index(self, index_path, doc_path):
        self.index = faiss.read_index(index_path)
        with open(doc_path, 'r') as f:
            self.documents = [line.strip() for line in f.readlines()]

    def search(self, query_embedding, top_n=5):
        D, I = self.index.search(query_embedding.cpu().numpy().reshape(1, -1), top_n)
        return [self.documents[i] for i in I[0]]

    def update_index(self, documents):
        embeddings = self.encode_documents(documents)
        self.add_embeddings(embeddings, documents)

#"""