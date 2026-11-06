import argparse
import os

_parser = argparse.ArgumentParser()
_parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "/home/vcl3d/coco_dataset_VOX"), help="COCO VOX dataset root (env: DATA_DIR)")
_parser.add_argument("--dest-dir", default=os.environ.get("DEST_DIR", "/home/vcl3d/coco_dataset_VOX_2000"), help="destination root (env: DEST_DIR)")
_args = _parser.parse_args()
import shutil
source_directory = os.path.join(_args.data_dir, "val2014_desc")
destination_directory = os.path.join(_args.dest_dir, "val2014_desc")
files = os.listdir(source_directory)
files = sorted(files)
num_files_to_copy = 5000
os.makedirs(destination_directory, exist_ok=True)
for i, file in enumerate(files):
    if i >= num_files_to_copy:
        break
    source_path = os.path.join(source_directory, file)
    destination_path = os.path.join(destination_directory, file)
    shutil.copy2(source_path, destination_path)
