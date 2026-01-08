import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch import Tensor
from torch.utils.data import DataLoader
import faiss
from .printer import log_info


class DealerAssistantModel:
    PRETRAINED_MODEL = "distiluse-base-multilingual-cased-v1"
    DEALER_ASSISTANT_MODEL = "models/dealer_assistant"

    def __init__(self, datasets: dict=None, embeddings=None):
        self._model = None
        self._index = None
        self._datasets = datasets
        self._embeddings = embeddings


    # ---- Getters and setters ----
    @property
    def datasets(self) -> dict | None:
        return self._datasets

    @datasets.setter
    def datasets(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError("datasets must be a dict")
        self._datasets = value

    @property
    def embeddings(self) -> list[Tensor] | np.ndarray | Tensor | dict[str, Tensor] | list[dict[str, Tensor]]:
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value) -> None:
        self._embeddings = value


    # ---- Implementation ----

    def train(self):
        triplets_df = self.datasets['triplets']
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

    def generate_embeddings(self, model_path: str=None):
        # model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        # model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        model_path = model_path if model_path else self.DEALER_ASSISTANT_MODEL
        model = SentenceTransformer(model_path)
        taken_df = self.datasets['works']

        taken_descriptions = taken_df['description'].dropna().tolist()

        # Generate embeddings for the descriptions
        # embeddings = model.encode(taken_descriptions, show_progress_bar=True)
        self.embeddings = model.encode(
            taken_descriptions,
            show_progress_bar=False,
            convert_to_numpy=True
        ).astype("float32")

        log_info("SBERT model embeddings generated.")

    def place_index(self):
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)

        dimension = self.embeddings.shape[1]

        # Cosine similarity via inner product
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(self.embeddings)

        log_info(
            f"FAISS index initialized with cosine similarity "
            f"({self._index.ntotal} vectors, dimension {dimension})."
        )

    def search(self, query: str, k: int=5) -> list[dict]:
        model = SentenceTransformer(self.DEALER_ASSISTANT_MODEL)

        query_embedding = model.encode([query])
        distances, indices = self._index.search(query_embedding, k)

        taken_df = self.datasets['works']

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

