from vllm import LLM, SamplingParams
import json
import multiprocessing
# from profiler.monitor import GPUMemoryMonitor
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # to prevent vllm from picking up string gpu id
import sys
from enum import enum

def main_function():
    def add_system_prompt(prompt):
        return prompt
        return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:"

    # read prompts from dataset
    prompts = []
    question_file = "./data/input/vicuna.jsonl" #args.question_file
    with open(question_file, "r") as f:
        for line in f:
            json_line = json.loads(line)
            if json_line["category"] not in ("counterfactual", "generic"):
                continue
            prompts.append(add_system_prompt(json_line["question"]))
    
    # set model parameters
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, min_tokens=100, max_tokens=1000)

    # select model
    llm = LLM(model="lmsys/vicuna-13b-v1.5", gpu_memory_utilization=0.9, disable_log_stats=False) #  Chaanan/vicuna-7b-v1.5-W8A8-Dynamic-Per-Token lmsys/vicuna-7b-v1.5
    # /home/chaanan/.cache/huggingface/hub/models--Chaanan--vicuna-7b-v1.5-W8A8-Dynamic-Per-Token/snapshots/d607e7f6393d17f42e546fa2827484d69de6dd29
    
    # generate
    outputs = llm.generate(prompts, sampling_params)

    num_input_tokens = 0
    num_output_tokens = 0

    # print outputs
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        num_input_tokens += len(prompt)
        num_output_tokens += len(generated_text)
    
    print("Num input words: ", num_input_tokens)
    print("Num output words: ", num_output_tokens)


if __name__ == "__main__":
    main_function()
