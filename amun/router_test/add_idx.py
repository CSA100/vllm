import json

def add_idx_to_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for idx, line in enumerate(infile):
            json_obj = json.loads(line)
            
            # Create a new dictionary with 'idx' first, followed by the original object
            new_obj = {'idx': idx}
            new_obj.update(json_obj)
            
            # Write the new object to the output file
            outfile.write(json.dumps(new_obj) + '\n')

# Example usage
input_file = './e2e_accuracy/wizard/control.jsonl'
output_file = './e2e_accuracy/wizard/control_x.jsonl'
add_idx_to_jsonl(input_file, output_file)
