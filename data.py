
# Install datasets library if needed
import subprocess
import sys

try:
    from datasets import load_dataset
    print("datasets library already installed")
except ImportError:
    print("Installing datasets library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
    from datasets import load_dataset

# Load the google/simpleqa-verified dataset
print("\nLoading google/simpleqa-verified dataset...")
dataset = load_dataset("google/simpleqa-verified")

# Display dataset structure
print("\nDataset splits:")
print(dataset)

# Get the column names from the dataset
print("\nColumn names:")
columns = dataset['simpleqa_verified'].column_names
for i, col in enumerate(columns, 1):
    print(f"{i}. {col}")

# Display column info with types
print("\nColumn details:")
print(dataset['simpleqa_verified'].features)

# Show first example to illustrate the data structure
print("\nFirst example:")
first_example = dataset['simpleqa_verified'][0]
for key, value in first_example.items():
    print(f"\n{key}: {value}")
