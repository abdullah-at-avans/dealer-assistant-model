from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # project root
# DATASET_PATH = "https://dev-icp.s3.eu-central-1.amazonaws.com/dataengineering/"
DATASET_PATH = (BASE_DIR / "datasets/")

DATA_FILES = {
    # "workorders": "workorders.csv",
    "messages": "remarks.csv",
    # "join": "join.csv",
    # "vehicles": "vehicles.csv",
    "works": "works.csv",
    "triplets": "triplets.csv",
}





