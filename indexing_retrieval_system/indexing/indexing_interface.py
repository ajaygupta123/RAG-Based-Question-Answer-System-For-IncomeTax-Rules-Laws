from abc import ABC, abstractmethod

class IndexingInterface(ABC):
    @abstractmethod
    def add_embeddings(self, embeddings, documents):
        pass

    @abstractmethod
    def save_index(self, index_path, doc_path):
        pass

    @abstractmethod
    def load_index(self, index_path, doc_path):
        pass

    @abstractmethod
    def search(self, query_embedding, top_n=5):
        pass

    @abstractmethod
    def update_index(self, documents):
        pass
