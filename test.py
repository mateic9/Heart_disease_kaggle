import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")
from datetime import datetime
MODEL_PATH = Path(os.getenv("MODEL_PATH"))
st = os.stat(MODEL_PATH)

date = datetime.fromtimestamp(st.st_mtime)

print(date)