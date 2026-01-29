import json
import os

# Define the folder containing the JSON files
folder_path = 'labelled_data/'  # Replace with the actual folder path
output_file = 'merged_data.json'  # Output file path

# Function to merge all JSON files from a folder with incremented 'id'
def merge_json_files_from_folder(folder_path, output_file):
    merged_data = []
    current_id = 1  # Start id from 1
    
    # Get all JSON files from the folder
    input_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    # Iterate through each file in the folder
    for file in input_files:
        file_path = os.path.join(folder_path, file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Update each item with a unique 'id'
                for item in data:
                    item['id'] = current_id
                    current_id += 1  # Increment id for the next item
                merged_data.extend(data)  # Append data from the file to merged_data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Remove the 'label' key from each entry
    for item in merged_data:
        if 'label' in item:
            del item['label']
    
    # Save the merged data to output file
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)
    print(f"Data successfully merged and saved to {output_file}")

# Merge the data
merge_json_files_from_folder(folder_path, output_file)