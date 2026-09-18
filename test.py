import os
from pathlib import Path

from datetime import datetime
MODEL_PATH = Path("models/heart_pipeline.joblib")
st = os.stat(MODEL_PATH)

date = datetime.fromtimestamp(st.st_mtime)

print(date)