from assistant.data_loader import load_datasets
from assistant.preprocessing import preprocess
from assistant.model import DealerAssistantModel
from assistant.printer import log_info
from assistant.timer import timed

def main():
    datasets = timed("Loading datasets", load_datasets)
    datasets = timed("Pre processing pipeline", preprocess, datasets)

    model = DealerAssistantModel(datasets)

    timed("Training", model.train)
    timed("Generate embeddings", model.generate_embeddings)
    timed("Saving model", model.save_model, 'mistralai/Mistral-Small-3.1-24B-Instruct-2503')

    log_info(model.embeddings.shape)

    log_info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
