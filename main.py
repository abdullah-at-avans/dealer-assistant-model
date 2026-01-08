from assistant.data_loader import load_datasets
from assistant.preprocessing import preprocess
from assistant.model import DealerAssistantModel
from assistant.printer import log_info
from assistant.timer import timed

def main():
    datasets = timed("Loading datasets", load_datasets)
    # datasets = timed("Pre processing pipeline", preprocess, datasets)

    model = DealerAssistantModel(datasets)

    # model_path = 'mistralai/Mistral-Small-3.1-24B-Instruct-2503'

    # timed("Training", model.train)
    # timed("Saving model", model.save_model)
    # timed("Generate embeddings", model.generate_embeddings)


    timed("Generate embeddings", model.generate_embeddings)
    timed("Placing serach index", model.place_index)

    timed("Seraching", model.search, 'Kan het Koel- en remvloeistof vervangen worden?')



if __name__ == "__main__":
    main()
