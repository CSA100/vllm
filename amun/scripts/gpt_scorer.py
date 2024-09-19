import argparse
from openai import OpenAI
client = OpenAI()
import json
from tqdm import tqdm

def create_completion(client, question, answer_a, answer_b):
    system_prompt = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."
    prompt = f"[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"


    completion = client.chat.completions.create(
        model=  "gpt-4o", # "gpt-3.5-turbo-0125", #
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

argsparser = argparse.ArgumentParser()
argsparser.add_argument("--flow1-file", type=str, required=True)
argsparser.add_argument("--flow2-file", type=str, required=True)
args = argsparser.parse_args()
flow1_file = args.flow1_file
flow2_file = args.flow2_file

with open(flow1_file, "r") as f:
    data_flow1 = [json.loads(line) for line in f]
with open(flow2_file, "r") as f:
    data_flow2 = [json.loads(line) for line in f]

# combine data and get scores
output_file = "amun_vs_control.jsonl"
with open(output_file, "w") as f:
    data = []
    for i in tqdm(range(len(data_flow1))):
        item = {}
        item["category"] = data_flow1[i]['category']
        item['prompt'] = data_flow1[i]['prompt']
        item['response_flow1'] = data_flow1[i]['response']
        item['response_flow2'] = data_flow2[i]['response']

        res = create_completion(client, item['prompt'], item['response_flow1'], item['response_flow2'])
        item['judgement_1'] = res
        item['score_1'] = res.split("[[")[1].split("]]")[0]

        res = create_completion(client, item['prompt'], item['response_flow2'], item['response_flow1'])
        item['judgement_2'] = res
        item['score_2'] = res.split("[[")[1].split("]]")[0]

        score1 = 0
        if item['score_1'] == "A":
            score1 = 1
        elif item['score_1'] == "B":
            score1 = -1

        score_2 = 0
        if item['score_2'] == "B":
            score_2 = 1
        elif item['score_2'] == "A":
            score_2 = -1

        combined_score = score1 + score_2
        final_score = 1 if combined_score > 0 else -1 if combined_score < 0 else 0
        item['final_score'] = final_score

        f.write(json.dumps(item) + "\n")
