import multiprocessing
import asyncio
import time
from vllm import LLM, SamplingParams
from vllm import AsyncLLMEngine, AsyncEngineArgs


class AsyncProcessor:
    def __init__(self, model_name: str, _gpu_memory_capacity: int, _gpu_memory_space: int):
        # Initialize multiprocessing components
        self.manager = multiprocessing.Manager()
        self._request_queue = multiprocessing.Queue()
        self._response_dict = self.manager.dict()
        
        # Initialize processor attributes
        self.gpu_mem_usage_rate = _gpu_memory_space / _gpu_memory_capacity
        self.request_count = 0
        self.model_name = model_name
        self.response_map = self.manager.dict()
        self.sampling_params = SamplingParams(
            temperature=0, 
            top_p=1, 
            logprobs=20, 
            prompt_logprobs=2,
            max_tokens=None
        )
        
        # Start processor in a separate process
        self._processor = multiprocessing.Process(
            target=self._async_processor_main
        )
        self._processor.start()
        
    def submit_request(self, prompt, request_id=None):
        """Submit a request and wait for its completion"""
        if request_id is None:
            request_id = str(time.time())
            
        # Submit the request
        self._request_queue.put((prompt, request_id))
        
        # Wait for completion with timeout
        timeout = 300  # 5 minutes timeout
        start_time = time.time()
        
        while request_id not in self._response_dict:
            if time.time() - start_time > timeout:
                print(f"Request {request_id} timed out")
                return None
            time.sleep(0.1)
        
        result = self._response_dict[request_id]
        return result

    def shutdown(self):
        """Gracefully shutdown the processor"""
        self._request_queue.put((None, None))
        self._processor.join()
        
    def _start_model(self):
        """Initialize the LLM engine"""
        args = AsyncEngineArgs(
                model=self.model_name,
                dtype="auto",
                enforce_eager=True,
                gpu_memory_utilization=0.8,
                swap_space=3,
                max_model_len=1024,
                kv_cache_dtype="auto",
                tensor_parallel_size=1,
                disable_log_requests=True,
                enable_prefix_caching=True,
            )
        args.disable_log_requests = True

        self.engine = AsyncLLMEngine.from_engine_args(
            args
        )
        
    async def _generate_request(self, prompt, request_id):
        """Internal method to generate response for a request"""
        print(prompt, request_id)
        try:
            results_generator = self.engine.generate(
                inputs=prompt,
                sampling_params=self.sampling_params,
                request_id=request_id,
            )
            print('ran this')
            
            results = [result async for result in results_generator]
            text_output = [output.text for output in results[-1].outputs]
            result_output = results[-1].outputs[0]
            
            with self.manager.Lock():
                #self.response_map[request_id] = text_output
                self.response_map[request_id] = result_output
                
            return result_output
        except Exception as e:
            print(f"Error processing request {request_id}: {str(e)}")
            return None
        
        
    async def _generate_request_text(self, prompt, request_id):
        """Internal method to generate response for a request"""
        try:
            results_generator = self.engine.generate(
                inputs=prompt,
                sampling_params=self.sampling_params,
                request_id=request_id,
            )
            
            results = [result async for result in results_generator]
            #text_output = [output.text for output in results[-1].outputs]
            text_output = results[-1].outputs[0].text
            
            with self.manager.Lock():
                self.response_map[request_id] = text_output
                
            return text_output
        except Exception as e:
            print(f"Error processing request {request_id}: {str(e)}")
            return None

    async def _process_requests(self):
        """Internal method to process requests from queue"""
        active_tasks = set()
        
        async def handle_task_completion(task, request_id):
            try:
                result = await task
                if result is not None:
                    self._response_dict[request_id] = result
                active_tasks.remove(task)
            except Exception as e:
                print(f"Task {request_id} failed: {str(e)}")
                active_tasks.remove(task)

        while True:
            try:
                while not self._request_queue.empty():
                    prompt, request_id = self._request_queue.get_nowait()
                    if prompt is None:
                        await asyncio.gather(*active_tasks)
                        return

                    task = asyncio.create_task(
                        self._generate_request(prompt, request_id)
                    )
                    active_tasks.add(task)
                    asyncio.create_task(handle_task_completion(task, request_id))

                if active_tasks:
                    await asyncio.sleep(0.1)
                else:
                    await asyncio.sleep(0.5)

            except multiprocessing.queues.Empty:
                if not active_tasks:
                    await asyncio.sleep(0.5)
                continue
            except Exception as e:
                print(f"Error in process_requests: {str(e)}")

    def _async_processor_main(self):
        """Internal method to run the async processor"""
        try:
            self._start_model()
            asyncio.run(self._process_requests())
        except Exception as e:
            print(f"Error in async_processor_main: {str(e)}")


if __name__ == "__main__":
    # Initialize the processor
    processor = AsyncProcessor(
        model_name="lmsys/vicuna-13b-v1.5",
        _gpu_memory_capacity=80,
        _gpu_memory_space=50
    )
    
    def process_prompt(prompt, idx):
        result = processor.submit_request(prompt, str(idx))
        print(f"Result for prompt {idx}: {result}")
        
    prompts = [
        "What is the capital of France?",
        "How do you make a chocolate cake?",
        "Why do leaves turn yellow in the fall?",
        "What's the fastest land animal?",
        "Explain the theory of relativity.",
        "How many planets are in our solar system?",
        "What year was the internet invented?",
        "Describe the water cycle to a child.",
        "Who wrote 'Pride and Prejudice'?",
        "How does photosynthesis work?",
        "What is the tallest mountain in the world?",
        "Translate 'hello' to Italian.",
        "What temperature does water boil at?",
        "Who was the first person on the moon?",
        "Name a book that won the Pulitzer Prize.",
        "What is the chemical formula for water?",
        "How many continents are there?",
        "What is the main ingredient in pizza dough?",
        "What is the population of Tokyo?",
        "How long does it take to fly from New York to London?"
    ]

    # Create and start processes
    processes = []
    for idx, prompt in enumerate(prompts):
        p = multiprocessing.Process(
            target=process_prompt,
            args=(prompt, idx)
        )
        processes.append(p)
        p.start()
        time.sleep(0.1)  # Small delay between starting processes

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Shutdown the processor
    processor.shutdown()