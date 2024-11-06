import json
from collections import OrderedDict

def add_idx_to_jsonl(input_file, output_file=None):
    modified_lines = []
    with open(input_file, 'r') as file:
        for idx, line in enumerate(file):
            # Parse the JSON line
            data = json.loads(line)
            
            # Create a new ordered dictionary with 'idx' at the front
            ordered_data = OrderedDict([('idx', idx)] + list(data.items()))
            
            # Append to the modified lines list
            modified_lines.append(ordered_data)
    
    # If an output file is specified, write modified lines to it
    if output_file:
        with open(output_file, 'w') as file:
            for item in modified_lines:
                file.write(json.dumps(item) + '\n')
    
    return modified_lines

# Usage example
input_file = '../data/input/combined.jsonl'
output_file = '../data/input/combined_idx.jsonl'
modified_data = add_idx_to_jsonl(input_file, output_file)

# Print the modified data
for item in modified_data:
    print(item)
