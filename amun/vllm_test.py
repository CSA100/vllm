from vllm import LLM, SamplingParams
import json
import multiprocessing
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # to prevent vllm from picking up string gpu id
os.environ["VLLM_ATTENTION_BACKEND"] = 'FLASHINFER' # use flashinfer backend. needed for kvc quantization.
import sys
import argparse
from collections import defaultdict
import subprocess
import re
import gc
# from llamaapi import LlamaAPI
import csv
from openai import OpenAI
openai_client = OpenAI()

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
                elif json_line['template_name'] == 'expansion_parallel':
                    prompt_templates['expansion_parallel'] = json_line['template']
        if len(prompt_templates) != 4:
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
        sampling_params = SamplingParams.from_optional(
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
        ) # max_tokens=1000
        if model_path == self.small_model_path:
            llm = LLM(model=model_path, gpu_memory_utilization=self.gpu_memory_utilization, disable_log_stats=False, enable_prefix_caching=True, kv_cache_dtype="fp8") #  Chaanan/vicuna-7b-v1.5-W8A8-Dynamic-Per-Token lmsys/vicuna-7b-v1.5
        else:
            llm = LLM(model=model_path, gpu_memory_utilization=self.gpu_memory_utilization, disable_log_stats=False, enable_prefix_caching=True) #  Chaanan/vicuna-7b-v1.5-W8A8-Dynamic-Per-Token lmsys/vicuna-7b-v1.5
        outputs = llm.generate(prompts, sampling_params)
        del llm
        gc.collect()
        return outputs


    def get_gpu_metrics(self, total_memory, vllm_memory_utilization, model_weight_size):
        # get gpu cache usage
        pattern = re.compile(
            r"Avg prompt throughput: [0-9.]+ tokens/s, "
            r"Avg generation throughput: [0-9.]+ tokens/s, "
            r"Running: [0-9]+ reqs, "
            r"Swapped: [0-9]+ reqs, "
            r"Pending: [0-9]+ reqs, "
            r"GPU KV cache usage: ([0-9.]+)%, "
            r"CPU KV cache usage: [0-9.]+%"
        )
        gpu_cache_usage = []
        with open('./metrics_output.log', 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    gpu_cache_usage.append(float(match.group(1)))
        total_memory *= vllm_memory_utilization
        total_cache_memory = total_memory - model_weight_size

        # compute metrics
        gpu_cache_usage = [x / 100.0 for x in gpu_cache_usage]
        peak_memory_kvc = max(gpu_cache_usage) * total_cache_memory
        peak_memory_total = peak_memory_kvc + model_weight_size
        total_time = len(gpu_cache_usage) * 0.1
        memory_time_integral_kvc = sum([x * total_cache_memory * 0.1 for x in gpu_cache_usage])
        memory_time_integral_total = sum([(x * total_cache_memory + model_weight_size) * 0.1 for x in gpu_cache_usage])

        return {
            "peak_memory_kvc": peak_memory_kvc,
            "peak_memory_total": peak_memory_total,
            "total_time": total_time,
            "memory_time_integral_kvc": memory_time_integral_kvc,
            "memory_time_integral_total": memory_time_integral_total
        }

    def extract_numbered_bullets(self, text):
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
            print('Running gpt-4-0613 for question:', question)

            # llama = LlamaAPI("LA-f800279eba404ce49b8c01d88be5c02d0d812d2f00ac445d995cda745a7dcc03")

            '''
            template = """
            [Question]
            [Insert user question here]

            [The Start of Assistant 1's Answer]
            [Insert Assistant 1's response here]
            [The End of Assistant 1's Answer]

            [The Start of Assistant 2's Answer]
            [Insert Assistant 2's response here]
            [The End of Assistant 2's Answer]

            [System]
            We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
            Please rate the helpfulness, relevance, accuracy, and level of detail of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.

            Please first output a single line containing only two values indicating the scores for Assistant 1 and Assistant 2, respectively. The two scores should be separated by a space.
            In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
            """
            '''

            system_prompt = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Be concise. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."

            prompt = f"[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"
            # api_request_json = {
            #     "model": "llama3.1-70b",
            #     "messages": [
            #         {"role": "system", "content": system_prompt},
            #         {"role": "user", "content": prompt}
            #     ],
            #     "max_tokens": 1200,
            #     "stream": False,
            # }
            completion = openai_client.chat.completions.create(
                model=  "gpt-4-0613", # "gpt-3.5-turbo-0125", #
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            judgement1 = completion.choices[0].message.content # llama.run(api_request_json).json()["choices"][0]["message"]["content"]
            print('Judgement1:', judgement1)
            score_1 = judgement1.split("[[")[1].split("]]")[0]

            prompt = f"[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_b}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_a}\n[The End of Assistant B's Answer]"
            # api_request_json = {
            #     "model": "llama3.1-70b",
            #     "messages": [
            #         {"role": "system", "content": system_prompt},
            #         {"role": "user", "content": prompt}
            #     ],
            #     "max_tokens": 1200,
            #     "stream": False,
            # }
            completion = openai_client.chat.completions.create(
                model=  "gpt-4o", # "gpt-3.5-turbo-0125", #
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            judgement2 = completion.choices[0].message.content  # llama.run(api_request_json).json()["choices"][0]["message"]["content"]
            print('Judgement2:', judgement2)
            score_2 = judgement2.split("[[")[1].split("]]")[0]

            score1 = 0
            if score_1 == "A":
                score1 = 1
            elif score_1 == "B":
                score1 = -1

            score2 = 0
            if score_2 == "B":
                score2 = 1
            elif score_2 == "A":
                score2 = -1

            combined_score = score1 + score2
            final_score = 1 if combined_score > 0 else -1 if combined_score < 0 else 0
            ans =  {
                "final_score": final_score,
                "judgement1": judgement1, 
                "judgement2": judgement2
            }
            return ans
        except Exception as e:
            print("error in getting accuracy:", e)
            ans =  {
                "final_score": "ERR",
                "judgement1": "ERR", 
                "judgement2": "ERR"
            }
            return ans

    def run_token_count(self):
        print(self.prompt_templates)
        for key, value in self.prompt_templates.items():
            print("Template:", key)
            output = self.generate(self.small_model_path, value)[0]
            num_input_tokens = len(output.prompt_token_ids)
            print("Num tokens:", num_input_tokens, "\n\n")

    def run(self):
        # initialize prompt queues
        standard_flow_prompts = []
        key_token_phase_prompts = []
        expansion_phase_prompts = []

        # augment standard flow and key token prompts
        with open("result.csv", mode="w", newline="") as csv_file:
            with open(self.prompt_file, "r") as f:
                i = 0
                for line in f:
                    i += 1
                    json_line = json.loads(line)
                    dataset = 'combined'
                    request_number = json_line["idx"]
                    standard_flow_prompt = self.embed_prompts(self.prompt_templates['standard'], [json_line["prompt"]])
                    key_token_prompt = self.embed_prompts(self.prompt_templates['key_token'], [json_line["prompt"]])

                    ### GET NUM REQUEST TOKENS
                    output = self.generate(self.small_model_path, json_line["prompt"])[0]
                    request_tokens = len(output.prompt_token_ids)

                    ### row detials
                    row_details = {
                        "dataset": dataset,
                        "request_number": request_number,
                        "request": json_line["prompt"],
                        "request_tokens": request_tokens
                    }

                    #### SM BASELINE
                    # clear out log file
                    with open('./metrics_output.log', 'w') as file:
                        pass

                    # get input and output lengths
                    output = self.generate(self.small_model_path, [standard_flow_prompt])[0]
                    num_input_tokens = len(output.prompt_token_ids)
                    num_output_tokens = len(output.outputs[0].token_ids)

                    # get metrics
                    metrics = self.get_gpu_metrics(80, 0.9, 6.5573)
                    throughput_tokens = float(num_output_tokens) / metrics["total_time"]
                    throughput_req = 1.0 / metrics["total_time"]

                    sm_baseline = {
                        "num_input_tokens": num_input_tokens,
                        "num_output_tokens": num_output_tokens,
                        "peak_memory_kvc": metrics["peak_memory_kvc"],
                        "peak_memory_total": metrics["peak_memory_total"],
                        "total_time": metrics["total_time"],
                        "memory_time_integral_kvc": metrics["memory_time_integral_kvc"],
                        "memory_time_integral_total": metrics["memory_time_integral_total"],
                        "throughput_tokens": throughput_tokens,
                        "throughput_req": throughput_req,
                        "input": standard_flow_prompt,
                        "output": output.outputs[0].text
                    }


                    #### LM BASELINE
                    # clear out log file
                    with open('./metrics_output.log', 'w') as file:
                        pass

                    # get input and output lengths
                    output = self.generate(self.large_model_path, [standard_flow_prompt])[0]
                    num_input_tokens = len(output.prompt_token_ids)
                    num_output_tokens = len(output.outputs[0].token_ids)

                    # get metrics
                    metrics = self.get_gpu_metrics(80, 0.9, 24.284)
                    throughput_tokens = float(num_output_tokens) / metrics["total_time"]
                    throughput_req = 1.0 / metrics["total_time"]

                    lm_baseline = {
                        "num_input_tokens": num_input_tokens,
                        "num_output_tokens": num_output_tokens,
                        "peak_memory_kvc": metrics["peak_memory_kvc"],
                        "peak_memory_total": metrics["peak_memory_total"],
                        "total_time": metrics["total_time"],
                        "memory_time_integral_kvc": metrics["memory_time_integral_kvc"],
                        "memory_time_integral_total": metrics["memory_time_integral_total"],
                        "throughput_tokens": throughput_tokens,
                        "throughput_req": throughput_req,
                        "input": standard_flow_prompt,
                        "output": output.outputs[0].text
                    }


                    #### LM-SM Key Token
                    # clear out log file
                    with open('./metrics_output.log', 'w') as file:
                        pass

                    # get input and output lengths
                    output = self.generate(self.large_model_path, [key_token_prompt])[0]
                    num_input_tokens = len(output.prompt_token_ids)
                    num_output_tokens = len(output.outputs[0].token_ids)

                    # get metrics
                    metrics = self.get_gpu_metrics(80, 0.9, 24.284)
                    throughput_tokens = float(num_output_tokens) / metrics["total_time"]
                    throughput_req = 1.0 / metrics["total_time"]
                    bullets = self.extract_numbered_bullets(output.outputs[0].text)

                    lm_sm_key_token = {
                        "num_input_tokens": num_input_tokens,
                        "num_output_tokens": num_output_tokens,
                        "peak_memory_kvc": metrics["peak_memory_kvc"],
                        "peak_memory_total": metrics["peak_memory_total"],
                        "total_time": metrics["total_time"],
                        "memory_time_integral_kvc": metrics["memory_time_integral_kvc"],
                        "memory_time_integral_total": metrics["memory_time_integral_total"],
                        "throughput_tokens": throughput_tokens,
                        "throughput_req": throughput_req,
                        "input": key_token_prompt,
                        "output": output.outputs[0].text,
                        "num_bullets": len(bullets)
                    }


                    #### LM-SM expansion
                    # clear out log file
                    with open('./metrics_output.log', 'w') as file:
                        pass

                    # create expansion prompt
                    expansion_prompt = self.embed_prompts(self.prompt_templates['expansion'], [json_line["prompt"], lm_sm_key_token["output"]])

                    # get input and output lengths
                    output = self.generate(self.small_model_path, [expansion_prompt])[0]
                    num_input_tokens = len(output.prompt_token_ids)
                    num_output_tokens = len(output.outputs[0].token_ids)

                    # get metrics
                    metrics = self.get_gpu_metrics(80, 0.9, 6.5573)
                    throughput_tokens = float(num_output_tokens) / metrics["total_time"]
                    throughput_req = 1.0 / metrics["total_time"]

                    lm_sm_expansion = {
                        "num_input_tokens": num_input_tokens,
                        "num_output_tokens": num_output_tokens,
                        "peak_memory_kvc": metrics["peak_memory_kvc"],
                        "peak_memory_total": metrics["peak_memory_total"],
                        "total_time": metrics["total_time"],
                        "memory_time_integral_kvc": metrics["memory_time_integral_kvc"],
                        "memory_time_integral_total": metrics["memory_time_integral_total"],
                        "throughput_tokens": throughput_tokens,
                        "throughput_req": throughput_req,
                        "input": expansion_prompt,
                        "output": output.outputs[0].text,
                        "num_parallel": 1
                    }

                    #### LM-SM expansion parallel
                    # clear out log file
                    with open('./metrics_output.log', 'w') as file:
                        pass

                    # create expansion prompt
                    expansion_prompts = []
                    for point, bullet in bullets:
                        expansion_prompts.append(self.embed_prompts(self.prompt_templates['expansion_parallel'], [json_line["prompt"], lm_sm_key_token["output"], point, point, bullet]))

                    # get input and output lengths
                    outputs = self.generate(self.small_model_path, expansion_prompts)
                    num_input_tokens = 0
                    num_output_tokens = 0
                    response = ""
                    for bullet, output in zip(bullets, outputs):
                        num_input_tokens += len(output.prompt_token_ids)
                        num_output_tokens += len(output.outputs[0].token_ids)
                        response += f"{bullet[0]}. {bullet[1]}" + output.outputs[0].text + "\n\n"

                    # # get metrics
                    metrics = self.get_gpu_metrics(80, 0.9, 6.5573)
                    throughput_tokens = float(num_output_tokens) / metrics["total_time"]
                    throughput_req = 1.0 / metrics["total_time"]

                    lm_sm_expansion_parallel = {
                        "num_input_tokens": num_input_tokens,
                        "num_output_tokens": num_output_tokens,
                        "peak_memory_kvc": metrics["peak_memory_kvc"],
                        "peak_memory_total": metrics["peak_memory_total"],
                        "total_time": metrics["total_time"],
                        "memory_time_integral_kvc": metrics["memory_time_integral_kvc"],
                        "memory_time_integral_total": metrics["memory_time_integral_total"],
                        "throughput_tokens": throughput_tokens,
                        "throughput_req": throughput_req,
                        "input": '\n\n'.join(expansion_prompts),
                        "output": response,
                        "num_parallel": len(expansion_prompts)
                    }


                    #### SM-SM Key Token
                    # clear out log file
                    with open('./metrics_output.log', 'w') as file:
                        pass

                    # get input and output lengths
                    output = self.generate(self.small_model_path, [key_token_prompt])[0]
                    num_input_tokens = len(output.prompt_token_ids)
                    num_output_tokens = len(output.outputs[0].token_ids)

                    # get metrics
                    metrics = self.get_gpu_metrics(80, 0.9, 6.5573)
                    throughput_tokens = float(num_output_tokens) / metrics["total_time"]
                    throughput_req = 1.0 / metrics["total_time"]
                    bullets = self.extract_numbered_bullets(output.outputs[0].text)

                    sm_sm_key_token = {
                        "num_input_tokens": num_input_tokens,
                        "num_output_tokens": num_output_tokens,
                        "peak_memory_kvc": metrics["peak_memory_kvc"],
                        "peak_memory_total": metrics["peak_memory_total"],
                        "total_time": metrics["total_time"],
                        "memory_time_integral_kvc": metrics["memory_time_integral_kvc"],
                        "memory_time_integral_total": metrics["memory_time_integral_total"],
                        "throughput_tokens": throughput_tokens,
                        "throughput_req": throughput_req,
                        "input": key_token_prompt,
                        "output": output.outputs[0].text,
                        "num_bullets": len(bullets)
                    }

                    #### SM-SM expansion
                    # clear out log file
                    with open('./metrics_output.log', 'w') as file:
                        pass

                    # create expansion prompt
                    expansion_prompt = self.embed_prompts(self.prompt_templates['expansion'], [json_line["prompt"], lm_sm_key_token["output"]])

                    # get input and output lengths
                    output = self.generate(self.small_model_path, [expansion_prompt])[0]
                    num_input_tokens = len(output.prompt_token_ids)
                    num_output_tokens = len(output.outputs[0].token_ids)

                    # get metrics
                    metrics = self.get_gpu_metrics(80, 0.9, 6.5573)
                    throughput_tokens = float(num_output_tokens) / metrics["total_time"]
                    throughput_req = 1.0 / metrics["total_time"]

                    sm_sm_expansion = {
                        "num_input_tokens": num_input_tokens,
                        "num_output_tokens": num_output_tokens,
                        "peak_memory_kvc": metrics["peak_memory_kvc"],
                        "peak_memory_total": metrics["peak_memory_total"],
                        "total_time": metrics["total_time"],
                        "memory_time_integral_kvc": metrics["memory_time_integral_kvc"],
                        "memory_time_integral_total": metrics["memory_time_integral_total"],
                        "throughput_tokens": throughput_tokens,
                        "throughput_req": throughput_req,
                        "input": expansion_prompt,
                        "output": output.outputs[0].text,
                        "num_parallel": 1
                    }

                    #### SM-SM expansion parallel
                    # clear out log file
                    with open('./metrics_output.log', 'w') as file:
                        pass

                    # create expansion prompt
                    expansion_prompts = []
                    for point, bullet in bullets:
                        expansion_prompts.append(self.embed_prompts(self.prompt_templates['expansion_parallel'], [json_line["prompt"], lm_sm_key_token["output"], point, point, bullet]))

                    # get input and output lengths
                    outputs = self.generate(self.small_model_path, expansion_prompts)
                    num_input_tokens = 0
                    num_output_tokens = 0
                    response = ""
                    for bullet, output in zip(bullets, outputs):
                        num_input_tokens += len(output.prompt_token_ids)
                        num_output_tokens += len(output.outputs[0].token_ids)
                        response += f"{bullet[0]}. {bullet[1]}\n\n" + output.outputs[0].text + "\n\n"

                    # # get metrics
                    metrics = self.get_gpu_metrics(80, 0.9, 6.5573)
                    throughput_tokens = float(num_output_tokens) / metrics["total_time"]
                    throughput_req = 1.0 / metrics["total_time"]

                    sm_sm_expansion_parallel = {
                        "num_input_tokens": num_input_tokens,
                        "num_output_tokens": num_output_tokens,
                        "peak_memory_kvc": metrics["peak_memory_kvc"],
                        "peak_memory_total": metrics["peak_memory_total"],
                        "total_time": metrics["total_time"],
                        "memory_time_integral_kvc": metrics["memory_time_integral_kvc"],
                        "memory_time_integral_total": metrics["memory_time_integral_total"],
                        "throughput_tokens": throughput_tokens,
                        "throughput_req": throughput_req,
                        "input": '\n\n'.join(expansion_prompts),
                        "output": response,
                        "num_parallel": len(expansion_prompts)
                    }

                    # # Add in accuracy results
                    # lm_sm_single_VS_lm = self.get_accuracy_results(json_line["prompt"], lm_sm_expansion["output"], lm_baseline["output"])
                    # lm_sm_single_VS_sm = self.get_accuracy_results(json_line["prompt"], lm_sm_expansion["output"], sm_baseline["output"])
                    # sm_sm_single_VS_lm = self.get_accuracy_results(json_line["prompt"], sm_sm_expansion["output"], lm_baseline["output"])
                    # sm_sm_single_VS_sm = self.get_accuracy_results(json_line["prompt"], sm_sm_expansion["output"], sm_baseline["output"])
                    # lm_sm_parallel_VS_lm = self.get_accuracy_results(json_line["prompt"], lm_sm_expansion_parallel["output"], lm_baseline["output"])
                    # lm_sm_parallel_VS_sm = self.get_accuracy_results(json_line["prompt"], lm_sm_expansion_parallel["output"], sm_baseline["output"])
                    # sm_sm_parallel_VS_lm = self.get_accuracy_results(json_line["prompt"], sm_sm_expansion_parallel["output"], lm_baseline["output"])
                    # sm_sm_parallel_VS_sm = self.get_accuracy_results(json_line["prompt"], sm_sm_expansion_parallel["output"], sm_baseline["output"])
                    # lm_VS_sm = self.get_accuracy_results(json_line["prompt"], lm_baseline["output"], sm_baseline["output"])

                    accuracy_results = {
                        # "lm_sm_single_VS_lm_final_score": lm_sm_single_VS_lm["final_score"],
                        # "lm_sm_single_VS_lm_judgement1": lm_sm_single_VS_lm["judgement1"],
                        # "lm_sm_single_VS_lm_judgement2": lm_sm_single_VS_lm["judgement2"],
                        # "lm_sm_single_VS_sm_final_score": lm_sm_single_VS_sm["final_score"],
                        # "lm_sm_single_VS_sm_judgement1": lm_sm_single_VS_sm["judgement1"],
                        # "lm_sm_single_VS_sm_judgement2": lm_sm_single_VS_sm["judgement2"],
                        # "sm_sm_single_VS_lm_final_score": sm_sm_single_VS_lm["final_score"],
                        # "sm_sm_single_VS_lm_judgement1": sm_sm_single_VS_lm["judgement1"],
                        # "sm_sm_single_VS_lm_judgement2": sm_sm_single_VS_lm["judgement2"],
                        # "sm_sm_single_VS_sm_final_score": sm_sm_single_VS_sm["final_score"],
                        # "sm_sm_single_VS_sm_judgement1": sm_sm_single_VS_sm["judgement1"],
                        # "sm_sm_single_VS_sm_judgement2": sm_sm_single_VS_sm["judgement2"],
                        # "lm_sm_parallel_VS_lm_final_score": lm_sm_parallel_VS_lm["final_score"],
                        # "lm_sm_parallel_VS_lm_judgement1": lm_sm_parallel_VS_lm["judgement1"],
                        # "lm_sm_parallel_VS_lm_judgement2": lm_sm_parallel_VS_lm["judgement2"],
                        # "lm_sm_parallel_VS_sm_final_score": lm_sm_parallel_VS_sm["final_score"],
                        # "lm_sm_parallel_VS_sm_judgement1": lm_sm_parallel_VS_sm["judgement1"],
                        # "lm_sm_parallel_VS_sm_judgement2": lm_sm_parallel_VS_sm["judgement2"],
                        # "sm_sm_parallel_VS_lm_final_score": sm_sm_parallel_VS_lm["final_score"],
                        # "sm_sm_parallel_VS_lm_judgement1": sm_sm_parallel_VS_lm["judgement1"],
                        # "sm_sm_parallel_VS_lm_judgement2": sm_sm_parallel_VS_lm["judgement2"],
                        # "sm_sm_parallel_VS_sm_final_score": sm_sm_parallel_VS_sm["final_score"],
                        # "sm_sm_parallel_VS_sm_judgement1": sm_sm_parallel_VS_sm["judgement1"],
                        # "sm_sm_parallel_VS_sm_judgement2": sm_sm_parallel_VS_sm["judgement2"],
                        # "lm_VS_sm_final_score": lm_VS_sm["final_score"],
                        # "lm_VS_sm_judgement1": lm_VS_sm["judgement1"],
                        # "lm_VS_sm_judgement2": lm_VS_sm["judgement2"]
                    }

                    # write to csv
                    dicts = [row_details, sm_baseline, lm_baseline, lm_sm_key_token, lm_sm_expansion, lm_sm_expansion_parallel, sm_sm_key_token, sm_sm_expansion, sm_sm_expansion_parallel, accuracy_results]
                    
                    # Prepare row data with proper escaping and formatting
                    final_row = []
                    headers = []
                    for d in dicts:
                        if i == 1:
                            headers.extend(d.keys())
                        # Format values appropriately for CSV/Excel
                        for value in d.values():
                            if isinstance(value, float):
                                # Format floats to 4 decimal places
                                formatted_value = f"{value:.4f}"
                            elif isinstance(value, str):
                                # Clean up strings - remove newlines and excessive spaces
                                formatted_value = value.replace('\n', ' ').replace('\r', ' ')
                                formatted_value = ' '.join(formatted_value.split())
                            else:
                                formatted_value = str(value)
                            final_row.append(formatted_value)

                    # Write to CSV with proper Excel-friendly settings
                    writer = csv.writer(csv_file, 
                                      delimiter=',',
                                      quotechar='"', 
                                      quoting=csv.QUOTE_MINIMAL,
                                      lineterminator='\n')
                    if i == 1:
                        writer.writerow(headers)
                    writer.writerow(final_row)

                    print(f'\n\nDONE: {i}\n\n')


                    


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

            # with open("./data/output/single_request/expansion_prompts.jsonl", "r") as f:
            #     for line in f:
            #         json_line = json.loads(line)
            #         expansion_phase_prompts.append(self.embed_prompts(self.prompt_templates['expansion'], [json_line["prompt"], json_line["key_tokens"]]))
            # outputs = self.generate(self.small_model_path, expansion_phase_prompts)
            # with open(f"{self.out_dir}_SM_expansion.jsonl", 'w') as jsonl_file:
            #     for output in outputs:
            #         jsonl_file.write(json.dumps({"prompt": output.prompt, "response": output.outputs[0].text}) + '\n')



        

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
    # batch_tester.run_token_count()


"""
    Example command
    python3 vllm_test.py --lmp="lmsys/vicuna-13b-v1.5" --smp="lmsys/vicuna-7b-v1.5" --mepp=20 --pf="./data/input/vicuna_g_cf.jsonl" --ptf="./data/prompt_templates.jsonl" --gpu_mem="0.9" --qkv="False" --out_dir="./data/output/"
    python3 vllm_test.py --lmp="/huggingface/models--lmsys--vicuna-13b-v1.5/snapshots/c8327bf999adbd2efe2e75f6509fa01436100dc2" --smp="/huggingface/models--Chaanan--vicuna-7b-v1.5-W8A8-Dynamic-Per-Token/snapshots/d607e7f6393d17f42e546fa2827484d69de6dd29" --mepp=20 --pf="./data/input/wizard.jsonl" --ptf="./data/prompt_templates.jsonl" --gpu_mem="0.9" --qkv="False" --out_dir="./data/output/"

    nohup python3 vllm_test.py --lmp="meta-llama/Llama-3.1-8B" --smp="meta-llama/Llama-3.2-1B-Instruct" --mepp=20 --pf="./data/input/routed/amun/combined.jsonl" --ptf="./data/prompt_templates.js
onl" --gpu_mem="0.9" --qkv="False" --out_dir="./data/output/" &

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
#             prompts.append(add_system_prompt(json_line["prompt"]))
    
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

