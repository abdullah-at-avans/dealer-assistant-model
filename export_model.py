from assistant import timed, load_datasets, DealerAssistantModel, preprocess

def main():
    datasets = timed("Loading datasets", load_datasets)
    datasets = timed("Pre processing pipeline", preprocess, datasets)

    model = DealerAssistantModel(datasets)

    timed("Training", model.train)
    timed("Saving model", model.save_model)

if __name__ == "__main__":
    main()
