from datasets import load_dataset

# Load the dataset
dataset = load_dataset("lemonilia/wikified_english_dictionary")

# Open file for writing
with open('simpleqa_verified_extract.txt', 'w', encoding='utf-8') as f:
    # Write header
    f.write("="*80 + "\n")
    f.write("SimpleQA Verified Dataset - Topic, Problem, Answer\n")
    f.write("="*80 + "\n\n")
    
    # Iterate through all examples
    for idx, example in enumerate(dataset['eval'], 1):
        f.write(f"{example['word']},{example['article']}\n")
        

print(f"✓ Exported {len(dataset['eval'])} entries to 'simpleqa_verified_extract.txt'")
print(f"✓ File size: {len(open('simpleqa_verified_extract.txt', 'r').read())} characters")

# Preview first 3 entries
print("\nPreview of first 3 entries:")
print("="*80)
for i in range(3):
    example = dataset['train'][i]
    print(f"{example['word']},{example['article']}")
    print("-"*80)
