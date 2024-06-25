from abc import ABC, abstractmethod

class RetrievalInterface(ABC):
    @abstractmethod
    def encode_documents(self, documents):
        pass

    @abstractmethod
    def encode_query(self, query):
        pass
