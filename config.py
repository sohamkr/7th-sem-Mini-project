
import os

DATA_TIMESTAMP_COL = "timestamp"
DATA_TARGET_COL = "target"

OUT_DIR = "outputs"
SEQ_LEN = 14
HORIZON = 1
BATCH_SIZE = 64
TEST_DAYS = 30
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)
