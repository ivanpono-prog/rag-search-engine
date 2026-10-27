from PIL import Image
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

class MultimodalSearch:
    def __init__(self, documents=None, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents if documents is not None else []

        self.texts = [
            f"{doc['title']}: {doc['description']}"
            for doc in self.documents
        ]

        self.text_embeddings = (
            self.model.encode(self.texts, show_progress_bar=True)
            if self.texts
            else None
        )

    def embed_image(self, image_path):
        image = Image.open(image_path)
        embedding = self.model.encode([image])[0]
        return embedding

    def search_with_image(self, image_path):
        image_embedding = self.embed_image(image_path)

        similarities = util.cos_sim(
            image_embedding,
            self.text_embeddings
        )[0]

        results = []

        for doc, similarity in zip(self.documents, similarities):
            results.append({
                **doc,
                "similarity": float(similarity)
            })

        results.sort(
            key=lambda result: result["similarity"],
            reverse=True
        )

        return results[:5]

def verify_image_embedding(image_path):
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)

    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path):
    from search_utils import load_movies

    documents = load_movies()
    search = MultimodalSearch(documents)

    return search.search_with_image(image_path)
