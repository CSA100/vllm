import json

def load_data_file(data_file):
    # Load the data.jsonl file into a dictionary keyed by some unique identifier
    data_map = {}
    with open(data_file, 'r') as dfile:
        for line in dfile:
            data_obj = json.loads(line.strip())
            # Assuming each object in data.jsonl has a unique identifier to match with the main file
            # For example, using a 'prompt' or 'id' as the unique key
            prompt_key = data_obj.get('question', None)  # Adjust this based on how prompts or identifiers match
            if prompt_key:
                # Store category and difficulty from the data file
                data_map[prompt_key] = {
                    'category': data_obj.get('Skill', ''),
                    'difficulty': data_obj.get('Difficulty', '')
                }
    return data_map

def extract_question_and_merge(input_file, data_file, output_file):
    # Load category and difficulty from the data.jsonl file
    data_map = load_data_file(data_file)

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            obj = json.loads(line.strip())
            
            # Extract the 'prompt' field
            prompt = obj.get('prompt', '')
            
            # Find the content between [Question] and [End of question]
            start_idx = prompt.find('[Question]')
            end_idx = prompt.find('[End of question]')
            
            question_content = None
            if start_idx != -1 and end_idx != -1:
                # Extract the content between [Question] and [End of question]
                question_content = prompt[start_idx + len('[Question]'):end_idx].strip()
                
                # Replace the original prompt with the extracted question
                obj['prompt'] = question_content
            
            prompt = question_content if question_content else prompt
            # Lookup category and difficulty from the data_map using the full prompt or another identifier
            if prompt in data_map:
                category = data_map[prompt].get('category', 'unknown')
                difficulty = data_map[prompt].get('difficulty', 'unknown')
            else:
                # Default values if no match found
                category = 'unknown'
                difficulty = 'unknown'
            
            # Create a new ordered dictionary to place category and difficulty at the front
            new_obj = {
                'category': category,
                'difficulty': difficulty
            }
            
            # Add all other original keys after category and difficulty
            new_obj.update(obj)
            
            # Write the modified object to the output file
            outfile.write(json.dumps(new_obj) + '\n')

# Example usage
input_file = './wizard/sm_raw.jsonl'    # Your input file (with questions to process)
data_file = './wizard.jsonl'      # The data file containing category and difficulty info
output_file = './wizard/sm.jsonl'  # Your output file
extract_question_and_merge(input_file, data_file, output_file)
