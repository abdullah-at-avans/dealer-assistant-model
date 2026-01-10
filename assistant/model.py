import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import faiss
from .printer import log_info
from pathlib import Path


class DealerAssistantModel:
    PRETRAINED_MODEL = "distiluse-base-multilingual-cased-v1"

    DEALER_ASSISTANT_MODEL = str(
        Path(__file__).resolve().parent.parent / "models" / "colab"
    )

    def __init__(self, datasets: dict=None, embeddings=None):
        self._model = None
        self._index = None
        self._datasets = datasets
        self._embeddings = embeddings

    # ---- Implementation ----
    def set_model(self, model_path: str) -> None:
        self._model = SentenceTransformer(model_path)
        log_info(f"SBERT model loaded from '{model_path}'.")

    def train(self):
        triplets_df = self._datasets['triplets']
        train_examples = [
            InputExample(texts=[row.anchor, row.positive, row.negative])
            for _, row in triplets_df.iterrows()
        ]

        # Load base model
        model = SentenceTransformer(self.PRETRAINED_MODEL)

        # DataLoader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=16
        )

        # Triplet Loss (cosine)
        train_loss = losses.TripletLoss(
            model=model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
        )

        # Train
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=2,
            warmup_steps=10,
            show_progress_bar=False
        )

        self._model = model
        log_info("SBERT model done fine tuning.")

    def save_model(self, save_to_path: str=None) -> None:
        save_to_path = save_to_path if save_to_path else self.DEALER_ASSISTANT_MODEL
        self._model.save(save_to_path)
        log_info(f"Model saved as '{save_to_path}'.")

    def generate_embeddings(self):
        taken_df = self._datasets['works']

        taken_descriptions = taken_df['description'].tolist()

        # Generate embeddings for the descriptions
        self._embeddings = self._model.encode(
            taken_descriptions,
            show_progress_bar=False,
            convert_to_numpy=True
        ).astype("float32")

        log_info("SBERT model embeddings generated.")

    def place_index(self):
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self._embeddings)

        dimension = self._embeddings.shape[1]

        # Cosine similarity via inner product
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(self._embeddings)

        log_info(
            f"FAISS index initialized with cosine similarity "
            f"({self._index.ntotal} vectors, dimension {dimension})."
        )

    def search(self, query: str, k: int=5) -> list[dict]:
        query_embedding = self._model.encode([query])
        distances, indices = self._index.search(query_embedding, k)

        taken_df = self._datasets['works']

        results = []
        for i, idx in enumerate(indices[0]):
            original_description = taken_df['description'].iloc[idx] # Assuming taken_df descriptions are in the same order as embeddings
            cid = taken_df['cid'].iloc[idx]
            results.append({
                'description': original_description,
                'cid': cid,
                'distance': distances[0][i]
            })
        log_info(results.__str__())
        return results

