import re
import csv

def extract_log_data(log_file_path, output_csv_path, filtered_csv_path):
    # Define a regex pattern to match the classification result portion of the log entry
    class_result_pattern = re.compile(r"classification result: {'predicted_class': (\d),.*?}")

    # Open the output CSV files for writing
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file, \
         open(filtered_csv_path, mode='w', newline='', encoding='utf-8') as filtered_csv_file:
        
        csv_writer = csv.writer(csv_file)
        filtered_csv_writer = csv.writer(filtered_csv_file)
        
        # Write the header row for both CSV files
        csv_writer.writerow(['idx', 'text', 'predicted_class'])
        filtered_csv_writer.writerow(['idx', 'text', 'predicted_class'])

        # Open the log file for reading
        with open(log_file_path, mode='r', encoding='utf-8') as log_file:
            text_accumulator = []  # To accumulate multi-line text entries
            idx = 0  # Initialize index counter
            
            for line in log_file:
                text_accumulator.append(line.strip())

                # Check if the current line contains the classification result
                match = class_result_pattern.search(line)
                if match:
                    # Join the accumulated text lines into a single line, and strip any leading/trailing whitespace
                    full_text = ' '.join(text_accumulator).replace('\n', ' ').replace('\r', ' ').strip()
                    predicted_class = match.group(1)
                    
                    # Extract the actual text portion before the classification result
                    text_match = re.search(r'text: (.*)', full_text)
                    if text_match:
                        text = text_match.group(1).strip()
                        
                        # Write the row to the first file with idx
                        csv_writer.writerow([idx, text, predicted_class])
                        
                        # Only write to the filtered file if predicted_class is not '1'
                        if predicted_class != '1':
                            filtered_csv_writer.writerow([idx, text, predicted_class])

                        idx += 1  # Increment index counter for the next row

                    # Clear the accumulator for the next log entry
                    text_accumulator = []

    print(f"Data extraction complete. Results saved to {output_csv_path} and {filtered_csv_path}")

# Specify the log file path and the output CSV file paths
log_file_path = './classifier_logs_wizard.log'
output_csv_path = 'output_with_idx_wizard.csv'
filtered_csv_path = 'output_filtered_wizard.csv'

# Run the log extraction function
extract_log_data(log_file_path, output_csv_path, filtered_csv_path)
