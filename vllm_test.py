from vllm import LLM, SamplingParams
import json
import multiprocessing
# from profiler.monitor import GPUMemoryMonitor
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # to prevent vllm from picking up string gpu id

def main_function(): 
    def add_system_prompt(prompt):
        return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:"

    # read prompts from dataset
    prompts = []
    question_file = "./vicuna.jsonl" #args.question_file
    with open(question_file, "r") as f:
        for line in f:
            json_line = json.loads(line)
            if json_line["category"] not in ("counterfactual", "generic"):
                continue
            prompts.append(add_system_prompt(json_line["question"]))
    
    # set model parameters
    sampling_params = SamplingParams(temperature=0, top_p=1, min_tokens=300, max_tokens=1000)

    # select model
    llm = LLM(model="lmsys/vicuna-7b-v1.5") #  Chaanan/vicuna-7b-v1.5-W8A8-Dynamic-Per-Token lmsys/vicuna-7b-v1.5
    
    # num_total_gpu = llm.llm_engine.cache_config.num_gpu_blocks
    # gpu_cache_usage_sys = 0.
    # if num_total_gpu is not None:
    #     num_free_gpu = sum(
    #         scheduler.block_manager.get_num_free_gpu_blocks()
    #         for scheduler in llm.llm_engine.scheduler)
    #     gpu_cache_usage_sys = 1.0 - (num_free_gpu / num_total_gpu)
    # print('before gpu cache usage: ', gpu_cache_usage_sys)

    
    # generate
    outputs = llm.generate(prompts, sampling_params)
    # with open('7b-quant-metrics.txt', 'w') as f:
    #     print(llm.llm_engine.get_metrics_history(), file=f)

    # num_total_gpu = llm.llm_engine.cache_config.num_gpu_blocks
    # gpu_cache_usage_sys = 0.
    # if num_total_gpu is not None:
    #     num_free_gpu = sum(
    #         scheduler.block_manager.get_num_free_gpu_blocks()
    #         for scheduler in llm.llm_engine.scheduler)
    #     gpu_cache_usage_sys = 1.0 - (num_free_gpu / num_total_gpu)
    # print('after gpu cache usage: ', gpu_cache_usage_sys)

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
    main_process = multiprocessing.Process(target=main_function)
    main_process.start()

    # Start GPU memory monitoring
    # gpu_monitor = GPUMemoryMonitor(pid=main_process.pid)
    # gpu_monitor.start_monitoring()

    # Wait for the main process to complete
    main_process.join()

    # Stop GPU memory monitoring
    # gpu_monitor.stop_monitoring()

    print("Monitoring process terminated.")
    # print(f"GPU memory and latency data saved to '{gpu_monitor.output_file}'.")
