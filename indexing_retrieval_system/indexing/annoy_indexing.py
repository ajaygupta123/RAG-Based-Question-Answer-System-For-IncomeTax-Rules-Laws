from indexing_interface import IndexingInterface
from annoy import AnnoyIndex

class AnnoyIndexing(IndexingInterface):
    def __init__(self, dimension, num_trees=10):
        self.index = AnnoyIndex(dimension, 'angular')
        self.dimension = dimension
        self.documents = []
        self.num_trees = num_trees
        self.counter = 0

    def add_embeddings(self, embeddings, documents):
        for embedding in embeddings:
            self.index.add_item(self.counter, embedding.cpu().numpy())
            self.documents.append(documents[self.counter])
            self.counter += 1

    def save_index(self, index_path, doc_path):
        self.index.build(self.num_trees)
        self.index.save(index_path)
        with open(doc_path, 'w') as f:
            for doc in self.documents:
                f.write(f"{doc}\n")

    def load_index(self, index_path, doc_path):
        self.index.load(index_path)
        with open(doc_path, 'r') as f:
            self.documents = [line.strip() for line in f.readlines()]

    def search(self, query_embedding, top_n=5):
        indices = self.index.get_nns_by_vector(query_embedding.cpu().numpy(), top_n)
        return [self.documents[i] for i in indices]

    def update_index(self, documents):
        embeddings = self.encode_documents(documents)
        self.add_embeddings(embeddings, documents)
