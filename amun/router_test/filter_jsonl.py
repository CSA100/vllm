import csv
import json

def get_filtered_indices(filtered_csv_path):
    """Reads the filtered CSV file and extracts the idx values."""
    filtered_indices = set()  # Use a set for fast lookup
    with open(filtered_csv_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            filtered_indices.add(int(row['idx']))
    return filtered_indices

def filter_jsonl_by_idx(input_jsonl_path, output_jsonl_path, filtered_indices):
    """Filters the JSONL file based on the filtered indices."""
    with open(input_jsonl_path, 'r', encoding='utf-8') as infile, open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            json_obj = json.loads(line)
            if json_obj['idx'] in filtered_indices:  # Only include the matching indices
                outfile.write(json.dumps(json_obj) + '\n')

# Example usage
filtered_csv_path = './output_filtered_wizard.csv'  # Path to the filtered CSV
input_jsonl_path = './e2e_accuracy/wizard/control.jsonl'  # Path to the indexed JSONL file
output_jsonl_path = './e2e_accuracy/wizard/control_filtered.jsonl'  # Output path for the filtered JSONL

# Get the indices from the filtered CSV
filtered_indices = get_filtered_indices(filtered_csv_path)

# Filter the JSONL file based on the indices
filter_jsonl_by_idx(input_jsonl_path, output_jsonl_path, filtered_indices)

print(f"Filtered JSONL file created: {output_jsonl_path}")
