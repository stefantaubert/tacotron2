import os
import shutil
import tempfile
from pathlib import Path

tmp_dir = "/tmp/test"
dir_path = "/tmp/datasets/THCHS"
content_dir = os.path.join(tmp_dir, "data_thchs30")
parent = Path(dir_path).parent
os.makedirs(parent, exist_ok=True)
dest = os.path.join(parent, "data_thchs30")
shutil.move(content_dir, dest)
os.rename(dest, dir_path)