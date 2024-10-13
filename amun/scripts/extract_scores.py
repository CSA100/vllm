import json
from collections import defaultdict

def count_final_scores_by_category(input_file):
    # Initialize a dictionary where each category maps to a dictionary of score counts
    category_scores = defaultdict(lambda: {'1': 0, '0': 0, '-1': 0})

    # Open and process the jsonl file
    with open(input_file, 'r') as infile:
        for line in infile:
            obj = json.loads(line.strip())
            
            # Extract the category and final_score from each entry
            category = obj.get('category', 'unknown')
            final_score = obj.get('final_score', None)
            
            # Check if final_score is one of 1, 0, or -1, and update counts accordingly
            if final_score in [1, 0, -1]:
                category_scores[category][str(final_score)] += 1

    return category_scores

def display_scores(category_scores):
    # Print the result
    for category, scores in category_scores.items():
        print(f"Category: {category}")
        print(f"  1: {scores['1']}")
        print(f"  0: {scores['0']}")
        print(f"  -1: {scores['-1']}")

# Example usage
input_file = '../data/output/wizard/evaluation/amun_vs_control.jsonl'  # Replace with your actual input file path
category_scores = count_final_scores_by_category(input_file)
display_scores(category_scores)
