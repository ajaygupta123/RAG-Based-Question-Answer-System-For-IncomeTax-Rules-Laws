class UnifiedRetrievalSystem:
    def __init__(self, retrieval_system, indexing_system):
        self.retrieval_system = retrieval_system
        self.indexing_system = indexing_system

    def index_documents(self, documents):
        embeddings = self.retrieval_system.encode_documents(documents)
        self.indexing_system.add_embeddings(embeddings, documents)

    def save_index(self, index_path, doc_path):
        self.indexing_system.save_index(index_path, doc_path)

    def load_index(self, index_path, doc_path):
        self.indexing_system.load_index(index_path, doc_path)

    def retrieve(self, query, top_n=5):
        query_embedding = self.retrieval_system.encode_query(query)
        return self.indexing_system.search(query_embedding, top_n)

    def update_index(self, documents):
        embeddings = self.retrieval_system.encode_documents(documents)
        self.indexing_system.update_index(documents)
