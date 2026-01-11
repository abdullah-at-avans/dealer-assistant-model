from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # project root

# DATASET_PATH = "https://dev-icp.s3.eu-central-1.amazonaws.com/dataengineering/" // remote path if datasets aren't locally
DATASET_PATH = (BASE_DIR / "datasets/")

# DATA_FILES = {
#     "workorders": "workorders.csv",
#     "messages": "remarks.csv",
#     "join": "join.csv",
#     "vehicles": "vehicles.csv",
#     "works": "works.csv",
#     "triplets": "triplets.csv",
# }

DATA_FILES = {
    "messages": "remarks.csv",
    "works": "works.csv",
    "triplets": "triplets.csv",
}
