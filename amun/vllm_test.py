from vllm import LLM, SamplingParams
import json
import multiprocessing
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # to prevent vllm from picking up string gpu id
import sys
import argparse
from collections import defaultdict
import subprocess

# from enum import enum

"""

    stitch function

    key point seperator function
    inputs:
        - degree of parellelization
    outputs:
        - seperated key points.

    run inference class
    inputs:
        - large model. 
        - small model. 
        - max parrellization. 
        - dataset questions file.
        - prompt template file.
        - gpu utilization
        - enable kv cache quantization (will set backed to flash infer)
    outputs (repeart for LM, SM and LM + SM):
        - overall metrics outfile
            - model size
            - total input len
            - total output len
            - model load time (kiv)
        - generation metrics outfile
            - time, Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 460.8 tokens/s, Running: 20 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 38.0%, CPU KV cache usage: 0.0%.
        - data outfile
            - prompt. response. prompt len. response len.

"""

class BatchTester:

    def __init__(self, large_model_path, small_model_path, max_expansion_phase_parallelization, prompt_file, prompt_template_file, gpu_memory_utilization, quantize_kv_cache, out_dir):
        self.large_model_path = large_model_path
        self.small_model_path = small_model_path
        self.max_expansion_phase_parallelization = max_expansion_phase_parallelization
        self.gpu_memory_utilization = gpu_memory_utilization
        self.quantize_kv_cache = quantize_kv_cache

        self.prompt_file = prompt_file
        self.prompt_template_file = prompt_template_file
        self.out_dir = out_dir

        self.prompt_templates = self.read_prompt_templates(prompt_template_file)

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
        if len(prompt_templates) != 3:
            raise Exception("Could not parse prompt templates")
        return prompt_templates


    def embed_prompts(self, prompt_template, prompts):
        """
        Embeds multiple prompts into the prompt_template at the placeholders {{prompt}}.
        
        :param prompt_template: A string containing multiple {{prompt}} placeholders.
        :param prompts: A list of prompts to embed into the template sequentially.
        :return: A string with the prompts embedded into the template.
        :raises ValueError: If the number of {{prompt}} placeholders and the number of prompts don't match.
        """
        # Count the number of {{prompt}} placeholders in the template
        placeholder_count = prompt_template.count("{{prompt}}")

        # Check if the number of prompts matches the number of placeholders
        if len(prompts) != placeholder_count:
            raise ValueError(f"Number of prompts ({len(prompts)}) does not match the number of placeholders ({placeholder_count}).")
        
        # Sequentially replace each {{prompt}} with the corresponding prompt from the list
        for prompt in prompts:
            prompt_template = prompt_template.replace("{{prompt}}", prompt, 1)
        
        return prompt_template

    def display_config(self):
        """Helper method to print the current configuration"""
        print("\n>>>>>> Batch Test Configuration >>>>>>")
        print(f"Large Model Path: {self.large_model_path}")
        print(f"Small Model Path: {self.small_model_path}")
        print(f"Max Expansion Phase Parallelization: {self.max_expansion_phase_parallelization}")
        print(f"Prompt File: {self.prompt_file}")
        print(f"Prompt Template File: {self.prompt_template_file}")
        print(f"GPU Memory Utilization: {self.gpu_memory_utilization}")
        print(f"Quantize KV Cache: {self.quantize_kv_cache}")
        print("<<<<<< Batch Test Configuration <<<<<<\n")

    def generate(self, model_path, prompts):
        sampling_params = SamplingParams(temperature=0.6, top_p=0.9, min_tokens=5, max_tokens=1000) # max_tokens=1000
        llm = LLM(model=model_path, gpu_memory_utilization=self.gpu_memory_utilization, disable_log_stats=False, enable_prefix_caching=True) #  Chaanan/vicuna-7b-v1.5-W8A8-Dynamic-Per-Token lmsys/vicuna-7b-v1.5
        outputs = llm.generate(prompts, sampling_params)
        return outputs

    def run(self):
        # initialize prompt queues
        standard_flow_prompts = []
        key_token_phase_prompts = []
        expansion_phase_prompts = []

        # augment standard flow and key token prompts
        with open(self.prompt_file, "r") as f:
            for line in f:
                json_line = json.loads(line)
                standard_flow_prompts.append(self.embed_prompts(self.prompt_templates['standard'], [json_line["question"]]))
                key_token_phase_prompts.append(self.embed_prompts(self.prompt_templates['key_token'], [json_line["question"]]))

        # small model
        # outputs = self.generate(self.small_model_path, standard_flow_prompts)
        # with open(f"{self.out_dir}_SM_standard.jsonl", 'w') as jsonl_file:
        #     for output in outputs:
        #         jsonl_file.write(json.dumps({"prompt": output.prompt, "response": output.outputs[0].text}) + '\n')

        # large model
        # outputs = self.generate(self.large_model_path, standard_flow_prompts)
        # with open(f"{self.out_dir}_LM_standard.jsonl", 'w') as jsonl_file:
        #     for output in outputs:
        #         jsonl_file.write(json.dumps({"prompt": output.prompt, "response": output.outputs[0].text}) + '\n')

        # lm key tokens
        # outputs = self.generate(self.large_model_path, key_token_phase_prompts)
        # with open(f"{self.out_dir}_LM_key_token.jsonl", 'w') as jsonl_file:
        #     for output in outputs:
        #         jsonl_file.write(json.dumps({"prompt": output.prompt, "response": output.outputs[0].text}) + '\n')

        # sm key tokens
        # outputs = self.generate(self.small_model_path, key_token_phase_prompts)
        # with open(f"{self.out_dir}_SM_key_token.jsonl", 'w') as jsonl_file:
        #     for output in outputs:
        #         jsonl_file.write(json.dumps({"prompt": output.prompt, "response": output.outputs[0].text}) + '\n')

        with open("./data/output/vicuna-20-q/expansion_prompts.jsonl", "r") as f:
            for line in f:
                json_line = json.loads(line)
                expansion_phase_prompts.append(self.embed_prompts(self.prompt_templates['expansion'], [json_line["prompt"], json_line["key_tokens"]]))
        outputs = self.generate(self.small_model_path, expansion_phase_prompts)
        with open(f"{self.out_dir}_SM_expansion.jsonl", 'w') as jsonl_file:
            for output in outputs:
                jsonl_file.write(json.dumps({"prompt": output.prompt, "response": output.outputs[0].text}) + '\n')



        

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Instantiate the BatchTester with various configurations.")

    # Add arguments for each parameter of the ModelManager class
    parser.add_argument("--lmp", type=str, required=True, help="Path to the large model file")
    parser.add_argument("--smp", type=str, required=True, help="Path to the small model file")
    parser.add_argument("--mepp", type=int, required=True, 
                        help="Maximum parallelization during the expansion phase")
    parser.add_argument("--pf", type=str, required=True, help="Path to the prompt file")
    parser.add_argument("--ptf", type=str, required=True, help="Path to the prompt template file")
    parser.add_argument("--gpu_mem", type=float, required=True, 
                        help="Percentage of GPU memory to use")
    parser.add_argument("--qkv", type=bool, required=True, 
                        help="Enable or disable quantization of key-value cache (True/False)")
    parser.add_argument("--out_dir", type=bool, required=True, 
                        help="Directory to store test outputs")

    # Parse arguments from the command line
    args = parser.parse_args()

    # Instantiate the BatchTester with parsed arguments
    batch_tester = BatchTester(
        large_model_path=args.lmp,
        small_model_path=args.smp,
        max_expansion_phase_parallelization=args.mepp,
        prompt_file=args.pf,
        prompt_template_file=args.ptf,
        gpu_memory_utilization=args.gpu_mem,
        quantize_kv_cache=args.qkv,
        out_dir=args.out_dir
    )

    batch_tester.run()


"""
    Example command
    python3 vllm_test.py --lmp="lmsys/vicuna-13b-v1.5" --smp="lmsys/vicuna-7b-v1.5" --mepp=20 --pf="./data/input/vicuna_g_cf.jsonl" --ptf="./data/prompt_templates.jsonl" --gpu_mem="0.9" --qkv="False" --out_dir="./data/output/"
    python3 vllm_test.py --lmp="lmsys/vicuna-13b-v1.5" --smp="/huggingface/models--Chaanan--vicuna-7b-v1.5-W8A8-Dynamic-Per-Token/snapshots/d607e7f6393d17f42e546fa2827484d69de6dd29" --mepp=20 --pf="./data/input/vicuna_g_cf.jsonl" --ptf="./data/prompt_templates.jsonl" --gpu_mem="0.9" --qkv="False" --out_dir="./data/output/"

    /home/chaanan/.cache/huggingface/hub/models--Chaanan--vicuna-7b-v1.5-W8A8-Dynamic-Per-Token/snapshots/d607e7f6393d17f42e546fa2827484d69de6dd29
    Chaanan/vicuna-7b-v1.5-W8A8-Dynamic-Per-Token 
    lmsys/vicuna-7b-v1.5    
"""


# def main_function():
#     def add_system_prompt(prompt):
#         return prompt
#         return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:"

#     # read prompts from dataset
#     prompts = []
#     question_file = "./data/input/vicuna.jsonl" #args.question_file
#     with open(question_file, "r") as f:
#         for line in f:
#             json_line = json.loads(line)
#             if json_line["category"] not in ("counterfactual", "generic"):
#                 continue
#             prompts.append(add_system_prompt(json_line["question"]))
    
#     # set model parameters
#     sampling_params = SamplingParams(temperature=0.6, top_p=0.9, min_tokens=100, max_tokens=1000)

#     # select model
#     llm = LLM(model="lmsys/vicuna-13b-v1.5", gpu_memory_utilization=0.9, disable_log_stats=False, enable_prefix_caching=True) #  Chaanan/vicuna-7b-v1.5-W8A8-Dynamic-Per-Token lmsys/vicuna-7b-v1.5    
#     # generate
#     outputs = llm.generate(prompts, sampling_params)

#     num_input_tokens = 0
#     num_output_tokens = 0

#     # print outputs
#     for output in outputs:
#         prompt = output.prompt
#         generated_text = output.outputs[0].text
#         print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
#         num_input_tokens += len(prompt)
#         num_output_tokens += len(generated_text)
    
#     print("Num input words: ", num_input_tokens)
#     print("Num output words: ", num_output_tokens)

