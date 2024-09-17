import json
import sys

def count_len(file_path):
    try:
        in_len = 0
        out_len = 0
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line.strip())
                    # Count words in the 'prompt' and 'response' fields
                    in_len += len(data['prompt'].split())
                    out_len += len(data['response'].split())
                except json.JSONDecodeError:
                    print(f"Invalid JSON in line: {line}")
                except KeyError:
                    print(f"Missing 'prompt' or 'response' field in line: {line}")
        print("The input word count is: ", in_len)
        print("The output word count is: ", out_len)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_jsonl_words.py <file.jsonl>")
        sys.exit(1)

    file_path = sys.argv[1]
    count_len(file_path)
