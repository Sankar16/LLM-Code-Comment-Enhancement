import os
import json
from sklearn.model_selection import train_test_split

# ---------- CONFIG ----------
INPUT_FOLDER = "labelled_data"   # put your path here
TRAIN_OUTPUT = "labelled_train.json"
TEST_OUTPUT = "labelled_test.json"
TRAIN_RATIO = 0.7
# ----------------------------

all_data = []

# Step 1: Load all JSON files and combine
for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".json"):
        file_path = os.path.join(INPUT_FOLDER, file)
        with open(file_path, "r") as f:
            data = json.load(f)

            # ensure data is a list
            if isinstance(data, dict):
                data = [data]

            all_data.extend(data)

# Step 2: Reassign global IDs
for idx, item in enumerate(all_data, start=1):
    item["id"] = str(idx)

# Step 3: Split 70–30 per label
train_set = []
test_set = []

# group by label
from collections import defaultdict
label_groups = defaultdict(list)

for item in all_data:
    label_groups[item["label"]].append(item)

# split each label group
for label, items in label_groups.items():
    train, test = train_test_split(items, train_size=TRAIN_RATIO, shuffle=True, random_state=42)
    train_set.extend(train)
    test_set.extend(test)

# Step 4: Save outputs
with open(TRAIN_OUTPUT, "w") as f:
    json.dump(train_set, f, indent=4)

with open(TEST_OUTPUT, "w") as f:
    json.dump(test_set, f, indent=4)

print("Done! Files created:", TRAIN_OUTPUT, TEST_OUTPUT)