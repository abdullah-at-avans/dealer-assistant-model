from assistant import timed, load_datasets, DealerAssistantModel, log_info, log_error, log_pass, log_warning


def main():
    datasets = timed("Loading datasets", load_datasets)

    model = DealerAssistantModel(datasets)

    timed("Generate embeddings", model.generate_embeddings)
    timed("Placing serach index", model.place_index)

    result: dict = timed("Seraching", model.search, 'Kan het Koel- en remvloeistof vervangen worden?')

    if result[0]['description'] == 'Remschijven incl. blokken vervangen (Voor)':
        log_pass("PASSED")
    else:
        log_error("Failed")

if __name__ == "__main__":
    main()
