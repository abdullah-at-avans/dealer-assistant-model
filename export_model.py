from assistant import timed, load_datasets, DealerAssistantModel, preprocess

def main():
    datasets = timed("Loading datasets", load_datasets)
    datasets = timed("Pre processing pipeline", preprocess, datasets)

    model = DealerAssistantModel(datasets)

    timed("Training", model.train)
    timed("Generate embeddings", model.generate_embeddings)

    timed("Placing search index", model.place_index)

    timed("Seraching", model.search, 'Kan het Koel- en remvloeistof vervangen worden?')


if __name__ == "__main__":
    main()
