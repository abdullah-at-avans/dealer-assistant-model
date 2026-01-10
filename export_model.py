from assistant import timed, load_datasets, DealerAssistantModel, preprocess, log_info


def main():
    datasets = timed("Loading datasets", load_datasets)
    datasets = timed("Pre processing pipeline", preprocess, datasets)

    model = DealerAssistantModel(datasets)

    timed("Training", model.train)

    timed("Saving model", model.save_model)

    log_info("Model has been exported, now testing the exported model...")

    timed("Generate embeddings", model.generate_embeddings)

    timed("Placing search index", model.place_index)

    timed("Single search", model.search, 'Kan het Koel- en remvloeistof vervangen worden?')


if __name__ == "__main__":
    main()
