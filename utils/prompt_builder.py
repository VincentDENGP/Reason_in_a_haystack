
import yaml
import random
import glob
from utils.utils import generate_random_number
from utils.tokenizer_utils import TokenizerHandler
import asyncio
import re

class Prompt:
    def __init__(self, context_lengths, config) -> None:
        self.config = config
        self.haystack_dir = self.config.haystack_dir
        self.n_needles_total = self.config.n_needles_total

        self.model_provider = self.config.model_provider
        self.model_name = self.config.model_name
        self.tokenizer = TokenizerHandler(self.model_name, self.model_provider)
        self.context_lengths = context_lengths
        self.final_context_length_buffer = self.config.final_context_length_buffer
        self.mode = self.config.mode
        self.needle = self.config.needle

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.tokenizer.estimate_token_length(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context
    
    def insert_range_needle(self, context, depth_percent, context_length, needles):

        def insert_at_point(tokens_context, tokens_needle, insertion_point):
            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            # period_tokens = self.tokenizer.encode_text_to_tokens('.')
            if "HF" in self.config.model_provider:
                period_tokens = self.tokenizer.encode_text_to_tokens('a.')
            else:
                period_tokens = self.tokenizer.encode_text_to_tokens('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]
            return tokens_new_context

        tokens_needle1 = self.tokenizer.encode_text_to_tokens(needles[0])
        tokens_needle2 = self.tokenizer.encode_text_to_tokens(needles[1])

        tokens_context = self.tokenizer.encode_text_to_tokens(context)
        context_length = len(tokens_context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle1) + len(tokens_needle2) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle1) - len(tokens_needle2)]
        if self.config.needles_dis >= context_length:
            tokens_new_context = tokens_needle1 + tokens_context + tokens_needle2
            new_context = self.tokenizer.decode_tokens(tokens_new_context)
            return new_context

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle2
            # print('decode sentence',self.decode_tokens(tokens_new_context))
            insertion_point = int(len(tokens_context) - self.config.needles_dis)
            # print('100 point',insertion_point)
            tokens_new_context = insert_at_point(tokens_new_context, tokens_needle1, insertion_point)
            # print('decode sentence',self.decode_tokens(tokens_new_context))

        else:
            # Go get the position (in terms of tokens) to insert your needle
            print('depth_percent',depth_percent)
            if depth_percent <= 50:
                print('<50')
                insertion_point = int(len(tokens_context) * (depth_percent / 100))
                print('point1:',insertion_point)
                tokens_new_context = insert_at_point(tokens_context, tokens_needle1, insertion_point)
                insertion_point = int(insertion_point + self.config.needles_dis)
                print('point2:',insertion_point)
                tokens_new_context = insert_at_point(tokens_new_context, tokens_needle2, insertion_point)
            else:
                print('>50')

                insertion_point = int(len(tokens_context) * (depth_percent / 100))
                print('point1:',insertion_point)
                tokens_new_context = insert_at_point(tokens_context, tokens_needle2, insertion_point)
                insertion_point = int(insertion_point - self.config.needles_dis)
                print('point2:',insertion_point)
                tokens_new_context = insert_at_point(tokens_new_context, tokens_needle1, insertion_point)
        # Convert back to a string and return it
        new_context = self.tokenizer.decode_tokens(tokens_new_context)
        return new_context
    
    def insert_needle(self, context, depth_percent, context_length, info, needle):


        tokens_needle = self.tokenizer.encode_text_to_tokens(needle)
        tokens_context = self.tokenizer.encode_text_to_tokens(context)

        context_length -= self.config.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            if "HF" in self.config.model_provider:
                period_tokens = self.tokenizer.encode_text_to_tokens('a.')
            else:
                period_tokens = self.tokenizer.encode_text_to_tokens('.')
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.tokenizer.decode_tokens(tokens_new_context)
        return new_context
    
    
    async def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily
        if self.config.pure_cal:
            print("use pure cal")
            def extract_info(line):
                # 定义要匹配的模式
                pattern = r'The special magic ([A-Za-z\s]+) number is: (\d+)\.'
                
                # 使用正则表达式查找匹配的内容
                match = re.search(pattern, line)
                
                if match:
                    # 提取匹配的词组、数字和原始句子
                    word_phrase = match.group(1)
                    number = match.group(2)
                    original_sentence = match.group(0)
                    return word_phrase, number, original_sentence
                else:
                    return None, None, None
            context = ""
            model_name = self.config.model_name.replace("/", "_")
            context_file_location = f'{model_name.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'
            needles_info = {}
            with open(self.config.load_prefix + f'/contexts/{context_file_location}_context.txt', 'r') as f:
                for line in f:
                    word_phrase, number, original_sentence = extract_info(line)
                    if word_phrase and number and original_sentence:
                        context += original_sentence
                        needles_info[word_phrase] = (number, 0)
            return context, needles_info


        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()


        # Insert your random statement according to your depth percent

        random_cities = random.sample(self.config.RANDOM_NEEDLE_CITIES, self.config.n_needles_total)

        needles_info = {}
        print("random_cities",random_cities)

        for random_city in random_cities:
            needles_info[random_city] = (
                str(generate_random_number(self.config.digits)),
                0 
            )
        print(needles_info)
        # inject needle without order limit
        if self.config.inject == 'range':
            needles = []
            for info in needles_info.items():
                random_city, (needle_rnd_number, _deptha_percent) = info
                needles.append(self.config.needle.format(city=random_city, rnd_number=needle_rnd_number))

            context = self.tokenizer.encode_and_trim(context, context_length)

            context = self.insert_range_needle(context, depth_percent, context_length, needles)

        else:
            needle=""
            for info in needles_info.items():
                random_city, (needle_rnd_number, _deptha_percent) = info
                needle += self.config.needle.format(city=random_city, rnd_number=needle_rnd_number)
            context = self.tokenizer.encode_and_trim(context, context_length - len(self.tokenizer.encode_text_to_tokens(needle)))
            context = self.insert_needle(context, depth_percent, context_length, info,needle)

        return context, needles_info

    def generate_prompt(self, context, needles_info):
        if self.mode == 'math':
            all_cities = list(needles_info.keys())
            suffix = ' + '.join(all_cities) + ' = ?'
            retrieval_question = self.config.retrieval_question + suffix
        elif self.mode == "normal":
            assert self.config.n_needles_total == 1, "n_needles_total must be 1 for mode 'normal'"
            all_cities = list(needles_info.keys())
            assert len(all_cities) == 1, "n_needles_total must be 1 for mode 'normal'"
            city = all_cities[0]
            retrieval_question = f"What is the special magic number of {city} according to the given context?"

        if self.model_provider == "Anthropic":
            with open(self.config.claude_prompt, 'r') as file:
                prompt = file.read()
            return prompt.format(retrieval_question=retrieval_question, context=context)
        elif self.model_provider == "OpenAI" or self.model_provider == "zhipu" or self.model_provider == "HF-chat":
            return [
                {
                    "role": "system",
                    "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct. [Output format]: city1 + city2 = result"
                },
                {
                    "role": "user",
                    "content": context
                },
                {
                    "role": "user",
                    "content": f"{retrieval_question} Only respond with the final result in single number without equation and explanation" #Don't give information outside the document or repeat your findings"
                }
            ]
        elif self.model_provider == "HF-instruct":
            try:
                p = "<s> [INST] " + context +'\n' + str(retrieval_question) +' !!Only respond the result and equation in format """cityname1 + cityname2 = final result""", DO NOT include explaination!! [/INST] </s>'
                return p
                with open(self.model_name+'.txt', 'r') as file:
                    prompt = file.read()
                return prompt.format(retrieval_question=retrieval_question, context=context)
            except:
                raise FileNotFoundError()
        
        
    async def generate_context_and_prompt(self, context_length, depth_percent):
        context, needles_info = await self.generate_context(context_length, depth_percent)
        prompt = self.generate_prompt(context, needles_info)
        return context, needles_info, prompt
    
