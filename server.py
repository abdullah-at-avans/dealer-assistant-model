from flask import Flask, request, jsonify
from assistant import *

app = Flask(__name__)

log_info("Starting server and loading model...")

datasets = timed("Loading datasets", load_datasets)

model = DealerAssistantModel(datasets)
timed("Setting model", model.set_model, model.DEALER_ASSISTANT_MODEL)

timed("Generate embeddings", model.generate_embeddings)
timed("Placing search index", model.place_index)

log_pass("Model is ready to receive requests.")


# -------------------- Routes

@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json(force=True)

        if not data or "inputs" not in data:
            log_warning("Invalid request: missing 'inputs'")
            return jsonify({"error": "Missing 'inputs' field"}), 400

        query = data["inputs"]
        k = data["k"]

        log_info("Received search request")

        results: list[dict] = model.search(query, k)
        results = to_json_safe(results)

        return jsonify({
            "query": query,
            "results": results
        })

    except Exception as e:
        log_error(str(e))
        return jsonify({"error": "Internal server error"}), 500

def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
