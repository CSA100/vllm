from vllm import LLM, SamplingParams
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # to prevent vllm from picking up string gpu id
import argparse
from collections import defaultdict
from openai import OpenAI
import csv
import re

class BatchTester:
    def __init__(self, large_model_path, small_model_path, prompt_file, prompt_template_file, out_dir):
        # Initialize models once
        self.llm_large = LLM(
            model=large_model_path,
            gpu_memory_utilization=0.7,
            disable_log_stats=True,
            enable_prefix_caching=True,
            max_model_len=2000
        )
        
        self.llm_small = LLM(
            model=small_model_path,
            gpu_memory_utilization=0.2,
            disable_log_stats=True,
            enable_prefix_caching=True,
            max_model_len=2000
        )
        
        self.prompt_file = prompt_file
        self.prompt_template_file = prompt_template_file
        self.out_dir = out_dir
        self.prompt_templates = self.read_prompt_templates(prompt_template_file)
        self.openai_client = OpenAI(api_key="")
        
        # Initialize sampling parameters
        self.sampling_params = SamplingParams.from_optional(
            max_tokens=512,
            stop=[
                "<|end_of_text|>", 
                "<|eot_id|>",
                "</s>",
                "<|im_end|>",
                "\nHuman:", 
                "\nAssistant:",
                "END"
            ],
        )

        self.display_config()

    def read_prompt_templates(self, prompt_template_file):
        prompt_templates = defaultdict(str)
        with open(prompt_template_file, "r") as f:
            for line in f:
                json_line = json.loads(line)
                if json_line['template_name'] == 'standard':
                    prompt_templates['standard'] = json_line['template']
                elif json_line['template_name'] == 'key_token':
                    prompt_templates['key_token'] = json_line['template']
                elif json_line['template_name'] == 'expansion':
                    prompt_templates['expansion'] = json_line['template']
                elif json_line['template_name'] == 'expansion_parallel':
                    prompt_templates['expansion_parallel'] = json_line['template']
        if len(prompt_templates) != 4:
            raise Exception("Could not parse prompt templates")
        return prompt_templates

    def embed_prompts(self, prompt_template, prompts):
        placeholder_count = prompt_template.count("{{prompt}}")
        if len(prompts) != placeholder_count:
            raise ValueError(f"Number of prompts ({len(prompts)}) does not match the number of placeholders ({placeholder_count}).")
        for prompt in prompts:
            prompt_template = prompt_template.replace("{{prompt}}", prompt, 1)
        return [
            {
                "role": "user",
                "content": prompt_template
            }
        ]

    def display_config(self):
        print("\n>>>>>> Batch Test Configuration >>>>>>")
        print(f"Prompt File: {self.prompt_file}")
        print(f"Prompt Template File: {self.prompt_template_file}")
        print("<<<<<< Batch Test Configuration <<<<<<\n")

    def extract_numbered_bullets(self, text):
        if not text.startswith('1.'):
            text = '1. ' + text
        pattern = r'\d+\.\s'
        parts = re.split(pattern, text)
        bullets = [part.strip() for part in parts if part.strip()]
        res = []
        for i in range(0, len(bullets)):
            res.append((str(i + 1), bullets[i]))
        return res

    def get_accuracy_results(self, question, answer_a, answer_b):
        try:
            print('Running gpt-4 for question:', question)
            system_prompt = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Be concise. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."

            prompt = f"[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"
            completion = self.openai_client.chat.completions.create(
                model="gpt-4-0613",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            judgement1 = completion.choices[0].message.content
            print('Judgement1:', judgement1)
            score_1 = judgement1.split("[[")[1].split("]]")[0]

            # Reverse order of answers
            prompt = f"[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_b}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_a}\n[The End of Assistant B's Answer]"
            completion = self.openai_client.chat.completions.create(
                model="gpt-4-0613",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            judgement2 = completion.choices[0].message.content
            print('Judgement2:', judgement2)
            score_2 = judgement2.split("[[")[1].split("]]")[0]

            score1 = 1 if score_1 == "A" else -1 if score_1 == "B" else 0
            score2 = 1 if score_2 == "B" else -1 if score_2 == "A" else 0
            final_score = 1 if score1 + score2 > 0 else -1 if score1 + score2 < 0 else 0

            return {
                "final_score": final_score,
                "judgement1": judgement1,
                "judgement2": judgement2
            }
        except Exception as e:
            print("Error in getting accuracy:", e)
            return {
                "final_score": "ERR",
                "judgement1": "ERR",
                "judgement2": "ERR"
            }

    def run(self):
        with open("result.csv", mode="w", newline="") as csv_file:
            with open(self.prompt_file, "r") as f:
                for i, line in enumerate(f, 1):
                    json_line = json.loads(line)
                    dataset = 'combined'
                    request_number = json_line["idx"]
                    
                    # Generate responses for different approaches
                    standard_flow_prompt = self.embed_prompts(self.prompt_templates['standard'], [json_line["prompt"]])
                    key_token_prompt = self.embed_prompts(self.prompt_templates['key_token'], [json_line["prompt"]])

                    # Get baseline outputs
                    sm_baseline = self.llm_small.chat(standard_flow_prompt, self.sampling_params)[0].outputs[0].text
                    lm_baseline = self.llm_large.chat(standard_flow_prompt, self.sampling_params)[0].outputs[0].text

                    # Get key token outputs
                    lm_key_token_output = self.llm_large.chat(key_token_prompt, self.sampling_params)[0].outputs[0].text
                    sm_key_token_output = self.llm_small.chat(key_token_prompt, self.sampling_params)[0].outputs[0].text

                    # Process expansions
                    lm_bullets = self.extract_numbered_bullets(lm_key_token_output)
                    sm_bullets = self.extract_numbered_bullets(sm_key_token_output)

                    # parallel expansion lm-sm
                    expansion_prompts = []
                    for point, bullet in lm_bullets:
                        expansion_prompts.append(self.embed_prompts(self.prompt_templates['expansion_parallel'], [json_line["prompt"], lm_key_token_output, point, point, bullet]))

                    outputs = self.llm_small.chat(expansion_prompts, self.sampling_params)
                    lm_sm_response = ""
                    for bullet, output in zip(lm_bullets, outputs):
                        lm_sm_response += f"{bullet[0]}. {bullet[1]}\n\n" + output.outputs[0].text + "\n\n"

                    # parallel expansion sm-sm
                    expansion_prompts = []
                    for point, bullet in sm_bullets:
                        expansion_prompts.append(self.embed_prompts(self.prompt_templates['expansion_parallel'], [json_line["prompt"], sm_key_token_output, point, point, bullet]))

                    outputs = self.llm_small.chat(expansion_prompts, self.sampling_params)
                    sm_sm_response = ""
                    for bullet, output in zip(sm_bullets, outputs):
                        sm_sm_response += f"{bullet[0]}. {bullet[1]}\n\n" + output.outputs[0].text + "\n\n"


                    # get accuracy results
                    lm_sm_vs_sm = self.get_accuracy_results(json_line["prompt"], lm_sm_response, sm_baseline)
                    lm_sm_vs_lm = self.get_accuracy_results(json_line["prompt"], lm_sm_response, lm_baseline)
                    sm_sm_vs_sm = self.get_accuracy_results(json_line["prompt"], sm_sm_response, sm_baseline)
                    sm_sm_vs_lm = self.get_accuracy_results(json_line["prompt"], sm_sm_response, lm_baseline)
                    lm_vs_sm = self.get_accuracy_results(json_line["prompt"], lm_baseline, sm_baseline)

                    # Write results
                    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    if i == 1:
                        headers = [
                            "dataset", "request_number", "request", 
                            "lm_baseline", "sm_baseline",
                            "lm_sm", "sm_sm",
                            "lm_sm_vs_sm_final_score", "lm_sm_vs_sm_judgement1", "lm_sm_vs_sm_judgement2",
                            "lm_sm_vs_lm_final_score", "lm_sm_vs_lm_judgement1", "lm_sm_vs_lm_judgement2",
                            "sm_sm_vs_sm_final_score", "sm_sm_vs_sm_judgement1", "sm_sm_vs_sm_judgement2",
                            "sm_sm_vs_lm_final_score", "sm_sm_vs_lm_judgement1", "sm_sm_vs_lm_judgement2",
                            "lm_vs_sm_final_score", "lm_vs_sm_judgement1", "lm_vs_sm_judgement2"
                        ]
                        writer.writerow(headers)

                    writer.writerow([
                        dataset,
                        request_number,
                        json_line["prompt"],
                        lm_baseline,
                        sm_baseline,
                        lm_sm_response,
                        sm_sm_response,
                        lm_sm_vs_sm["final_score"], lm_sm_vs_sm["judgement1"], lm_sm_vs_sm["judgement2"],
                        lm_sm_vs_lm["final_score"], lm_sm_vs_lm["judgement1"], lm_sm_vs_lm["judgement2"],
                        sm_sm_vs_sm["final_score"], sm_sm_vs_sm["judgement1"], sm_sm_vs_sm["judgement2"],
                        sm_sm_vs_lm["final_score"], sm_sm_vs_lm["judgement1"], sm_sm_vs_lm["judgement2"],
                        lm_vs_sm["final_score"], lm_vs_sm["judgement1"], lm_vs_sm["judgement2"]
                    ])

                    print(f'\n\nDONE: {i}\n\n')
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch testing with persistent LLM instances")
    parser.add_argument("--lmp", type=str, required=True, help="Path to the large model file")
    parser.add_argument("--smp", type=str, required=True, help="Path to the small model file")
    parser.add_argument("--pf", type=str, required=True, help="Path to the prompt file")
    parser.add_argument("--ptf", type=str, required=True, help="Path to the prompt template file")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to store test outputs")

    args = parser.parse_args()
    
    batch_tester = BatchTester(
        large_model_path=args.lmp,
        small_model_path=args.smp,
        prompt_file=args.pf,
        prompt_template_file=args.ptf,
        out_dir=args.out_dir
    )

    batch_tester.run()


"""""
    python3 acc_test.py --lmp="meta-llama/Llama-3.1-8B-Instruct" --smp="neuralmagic/Llama-3.2-1B-Instruct-quantized.w8a8" --pf="./data/input/routed/amun/combined.jsonl" --ptf="./data/prompt_templates.jsonl" --out_dir="./data/output/"
"""""