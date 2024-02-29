from datetime import datetime, timezone
from asyncio import Semaphore
import asyncio
import time
import os
import json
from anthropic import AsyncAnthropic, Anthropic
from openai import AsyncOpenAI

from utils.tokenizer_utils import TokenizerHandler
from utils.prompt_builder import Prompt
from utils.parser import evaluate_math_response


from zhipuai import ZhipuAI
class LLMNeedleHaystackTester:
    def __init__(self, config):
        self.config = config
        self.tokenizer = TokenizerHandler(model_name=config.model_name, model_provider=config.model_provider)
        self.prompter = Prompt(config.context_lengths,config)
        if self.config.model_provider == "OpenAI":
            self.model_to_test = AsyncOpenAI(api_key=self.config.openai_api_key, base_url = self.config.url )
        elif self.config.model_provider == "Anthropic":
            self.model_to_test = AsyncAnthropic(api_key=self.config.anthropic_api_key)
        elif self.config.model_provider == "zhipu":
            self.model_to_test = ZhipuAI(api_key=self.config.zhipu_api_key)
        elif "HF" in self.config.model_provider:
            self.model_to_test = AsyncOpenAI(api_key=self.config.openai_api_key, base_url = self.config.url )



    async def bound_evaluate_and_log(self, sem, *args):
        async with sem:
            await self.evaluate_and_log(*args)

    async def run_test(self):
        sem = Semaphore(self.config.num_concurrent_requests)
        tasks = []
        for context_length in self.config.context_lengths:
            for depth_percent in self.config.document_depth_percents:
                tasks.append(self.bound_evaluate_and_log(sem, context_length, depth_percent))
        await asyncio.gather(*tasks)

    async def evaluate_and_log(self, context_length, depth_percent):
        if self.result_exists(context_length, depth_percent):
            return
        context, needles_info = await self.prompter.generate_context(context_length, depth_percent)
        prompt = self.prompter.generate_prompt(context, needles_info)
        # print("prompt",prompt)
        test_start_time = time.time()
        if self.config.model_provider == "OpenAI" or self.config.model_provider == "HF-chat":
            async def send_request(self, prompt):
                max_attempts = 30
                for attempt in range(1, max_attempts + 1):
                    try:
                        if self.config.model_provider != "zhipu":
                            tem = 0
                        else:
                            tem = 0.01
                        response = await self.model_to_test.chat.completions.create(
                            model=self.config.model_name,
                            messages=prompt,
                            max_tokens=300,
                            temperature=tem
                        )
                        print(f"Attempt {attempt}: Success")
                        response = response.choices[0].message.content
                        return response  # 如果请求成功，返回响应并结束循环
                    except Exception as e:  # 捕获任何异常
                        print(f"Attempt {attempt}: An error occurred - {e}")
                        if attempt == max_attempts:
                            print("Reached the maximum number of attempts. Aborting.")
                            # raise  # 如果达到最大尝试次数，重新抛出异常
                            pass
                        await asyncio.sleep(1)  # 简单的延迟，防止立即重试，这里设置为1秒
                return -99999

            response = await send_request(self, prompt)

        elif self.config.model_provider == "HF-instruct":

            async def send_request(self, prompt):
                max_attempts = 30
                for attempt in range(1, max_attempts + 1):
                    try:
                        # print("prompt",prompt)
                        response = await self.model_to_test.completions.create(
                            model=self.config.model_name,
                            prompt=prompt,
                            max_tokens=50,
                            temperature=0
                        )
                        print(f"Attempt {attempt}: Success with {response}")
                        print('model answer:',response.choices[0].text)
                        # response = response.choices[0].message.content
                        response = response.choices[0].text
                        return response  # 如果请求成功，返回响应并结束循环
                    except Exception as e:  # 捕获任何异常
                        print(f"Attempt {attempt}: An error occurred - {e}")
                        if attempt == max_attempts:
                            print("Reached the maximum number of attempts. Aborting.")
                            # raise  # 如果达到最大尝试次数，重新抛出异常
                            pass
                        await asyncio.sleep(1)  # 简单的延迟，防止立即重试，这里设置为1秒
                return -99999

            response = await send_request(self, prompt)
        elif self.config.model_provider == "zhipu":
            def send_request(self, prompt):
                max_attempts = 30
                for attempt in range(1, max_attempts + 1):
                    try:
                        if self.config.model_provider != "zhipu":
                            tem = 0
                        else:
                            tem = 0.01
                        response = self.model_to_test.chat.completions.create(
                            model=self.config.model_name,
                            messages=prompt,
                            max_tokens=300,
                            temperature=tem
                        )
                        print(f"Attempt {attempt}: Success")
                        response = response.choices[0].message.content
                        return response  # 如果请求成功，返回响应并结束循环
                    except Exception as e:  # 捕获任何异常
                        print(f"Attempt {attempt}: An error occurred - {e}")
                        if attempt == max_attempts:
                            print("Reached the maximum number of attempts. Aborting.")
                            # raise  # 如果达到最大尝试次数，重新抛出异常
                            pass
                        time.sleep(1)  # 简单的延迟，防止立即重试，这里设置为1秒
                return -99999
            response = send_request(self, prompt)

        elif self.config.model_provider == "Anthropic":
            response = await self.model_to_test.completions.create(
                model=self.config.model_name,
                max_tokens_to_sample=300,
                prompt=prompt,
                temperature=0
            )
            response = response.completion

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        score1, score2, score3, parse1, parse2, parse3 = evaluate_math_response(response, needles_info)
        results = {
            'model' : self.config.model_to_test_description,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.config.results_version,
            'needle' : self.config.needle,
            'model_response' : response,
            'score1' : score1,
            'score2' : score2,
            'score3' : score3,
            'parse1' : parse1,
            'parse2' : parse2,
            'parse3' : parse3,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'),
            'needles_info' : needles_info
        }
        self.config.testing_results.append(results)
        if self.config.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score1: {score1}")
            print (f"Score2: {score2}")
            print (f"Score3: {score3}")
            print (f"Response: {response}\n")
        model_name = self.config.model_name.replace("/", "_")
        context_file_location = f'{model_name.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'
        if self.config.save_contexts:
            results['file_name'] : context_file_location

            # Save the context to file for retesting
            if not os.path.exists(self.config.save_prefix + '/contexts'):
                os.makedirs(self.config.save_prefix + '/contexts')

            with open(self.config.save_prefix + f'/contexts/{context_file_location}_context.txt', 'w') as f:
                f.write(context)
        if self.config.save_results:
            if not os.path.exists(self.config.save_prefix + '/results'):
                os.makedirs(self.config.save_prefix + '/results')

            # Save the result to file for retesting
            with open(self.config.save_prefix + f'/results/{context_file_location}_results.json', 'w') as f:
                json.dump(results, f)
        if self.config.seconds_to_sleep_between_completions:
            await asyncio.sleep(self.config.seconds_to_sleep_between_completions)

    def result_exists(self, context_length, depth_percent):
        results_dir = 'results/'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        for filename in os.listdir(results_dir):
            if filename.startswith(self.config.save_prefix) and filename.endswith('.json'):
                parts = filename.split('_')
                if len(parts) == 6:
                    model_name = parts[1]
                    length = parts[2]
                    depth = parts[3]
                    if model_name == self.config.model_name.replace(".", "_") and length == str(context_length) and depth == str(int(depth_percent*100)):
                        return True
        return False

    def get_results(self):
        return self.testing_results

    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.config.model_name}")
        print (f"- Context Lengths: {len(self.config.context_lengths)}, Min: {min(self.config.context_lengths)}, Max: {max(self.config.context_lengths)}")
        print (f"- Document Depths: {len(self.config.document_depth_percents)}, Min: {min(self.config.document_depth_percents)}%, Max: {max(self.config.document_depth_percents)}%")
        print (f"- Needle: {self.config.needle.strip()}")
        print ("\n\n")

    def start_test(self):
        if self.config.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test())