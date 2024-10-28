import csv
import json
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert CSV to two JSONL files based on predicted_class.')
    parser.add_argument('input_csv', help='Path to the input CSV file.')
    parser.add_argument('--output_class_0', default='class_0.jsonl', help='Output JSONL file for predicted_class 0.')
    parser.add_argument('--output_class_1', default='class_1.jsonl', help='Output JSONL file for predicted_class 1.')
    return parser.parse_args()

def validate_csv_headers(headers):
    required_headers = {'idx', 'text', 'predicted_class'}
    if not required_headers.issubset(set(headers)):
        missing = required_headers - set(headers)
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")

def process_csv(input_csv, output_class_0, output_class_1):
    with open(input_csv, mode='r', encoding='utf-8') as csvfile, \
         open(output_class_0, mode='w', encoding='utf-8') as jsonl0, \
         open(output_class_1, mode='w', encoding='utf-8') as jsonl1:
        
        reader = csv.DictReader(csvfile)
        validate_csv_headers(reader.fieldnames)
        
        for row_number, row in enumerate(reader, start=2):  # Start at 2 to account for header
            try:
                idx = row['idx']
                text = row['text']
                predicted_class = row['predicted_class']
                
                # Convert predicted_class to integer
                predicted_class = int(predicted_class)
                
                json_entry = {
                    "idx": idx,
                    "prompt": text
                }
                
                if predicted_class == 0:
                    jsonl0.write(json.dumps(json_entry) + '\n')
                elif predicted_class == 1:
                    jsonl1.write(json.dumps(json_entry) + '\n')
                else:
                    print(f"Warning: Row {row_number} has an unexpected predicted_class '{predicted_class}'. Skipping.")
            except ValueError:
                print(f"Warning: Row {row_number} has invalid data. Skipping.")
            except KeyError as e:
                print(f"Warning: Row {row_number} is missing column {e}. Skipping.")

def main():
    args = parse_arguments()
    
    # Check if input CSV exists
    if not os.path.isfile(args.input_csv):
        print(f"Error: The file '{args.input_csv}' does not exist.")
        return
    
    try:
        process_csv(args.input_csv, args.output_class_0, args.output_class_1)
        print(f"Successfully created '{args.output_class_0}' and '{args.output_class_1}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
