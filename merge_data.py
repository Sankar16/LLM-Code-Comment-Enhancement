import json
import os
from glob import glob

# Folder containing your JSON files
INPUT_FOLDER = "labelled_data"
OUTPUT_FILE = "labelled_data.json"

def merge_and_renumber_json(folder_path, output_file):
    all_items = []
    current_id = 1

    # Read all .json files in folder
    for file_path in sorted(glob(os.path.join(folder_path, "*.json"))):
        with open(file_path, "r") as f:
            data = json.load(f)

        # Update IDs
        for item in data:
            item["id"] = str(current_id)
            current_id += 1
            all_items.append(item)

    # Save merged file
    with open(output_file, "w") as out:
        json.dump(all_items, out, indent=4)

    print(f"Merged {current_id - 1} items into {output_file}")

# Run the function
merge_and_renumber_json(INPUT_FOLDER, OUTPUT_FILE)