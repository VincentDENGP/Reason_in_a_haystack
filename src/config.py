from datetime import datetime
from dotenv import load_dotenv
import os
import tiktoken
from anthropic import AsyncAnthropic, Anthropic
import numpy as np
from openai import AsyncOpenAI
from langchain.chat_models import ChatOpenAI

class LLMNeedleHaystackTesterArgs:
    def __init__(self,
            needle="\nThe special magic {city} number is: {rnd_number}.\n",
            haystack_dir="PaulGrahamEssays",
            retrieval_question="What is the result of the following math problem?",
            results_version = 1,
            context_lengths_min = 1000,
            context_lengths_max = 128000,
            context_lengths_num_intervals = 35,
            context_lengths = None,
            document_depth_percent_min = 0,
            document_depth_percent_max = 100,
            document_depth_percent_intervals = 35,
            document_depth_percents = None,
            document_depth_percent_interval_type = "linear",
            model_provider = "OpenAI",
            openai_api_key= "EMPTY", 
            anthropic_api_key = None,
            zhipu_api_key = None,
            model_name='gpt-4-turbo-preview',
            url = 'https://api.openai.com',
            num_concurrent_requests = 1,
            save_results = True,
            save_contexts = True,
            final_context_length_buffer = 400,
            seconds_to_sleep_between_completions = None,
            print_ongoing_status = True,
            n_needles_total = 2,
            n_needles_retrieve = -1,
            needles_dis = 1000,
            save_prefix = '',
            inject = '',
            mode = 'math',
            pure_cal = False,
            load_prefix = "",
            digits = 2,
            claude_prompt = "prompts/Anthropic_prompt.txt"
            ):
        """        
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic' or 'HF'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param zhipu_api_key: The API key for Zhipu. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        :param n_needles_total: The number of needles to be found in the haystack. Default is 2.
        :param n_needles_retrieve: deprecated
        :param needles_dis: the distance between the needles
        :param save_prefix: the prefix of the save path
        :param inject: when you leave it blank two needle will be concat and inject, to inject with distance = needles_dis, you can set it to 'range'
        :param mode: the mode of the test, default is math: calculate the addition of the two needles
        :pure_cal: if you want to test the pure math, set it to True, and pass the initizal logging path
        :load_prefix: path to load the previous Reason in the Haystack result
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        print('provider',self.model_provider)
        self.testing_results = []
        self.mode = mode
        self.url = url

        # match needle
        self.n_needles_total = n_needles_total 
        self.n_needles_retrieve = n_needles_retrieve

        # range needle
        self.needles_dis = needles_dis
        self.inject = inject
        self.context_lengths_min = context_lengths_min

        #pure math
        self.pure_cal = pure_cal
        self.load_prefix = load_prefix

        self.digits = digits
        self.claude_prompt = claude_prompt

        # Get today's date
        today = datetime.now()

        # Format the date as 'xx-xx-xx' (e.g., '21-02-24' for February 24, 2021)
        formatted_date = today.strftime('%y-%m-%d')
        self.model_name = model_name
        if '/' in model_name:
            model_name = model_name.replace('/','_')
            
        self.save_prefix = save_prefix + model_name + '_' +str(context_lengths_min) + '_' + str(context_lengths_max) + '_' + str(formatted_date) + '_'

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths
        print("self.context_lengths",self.context_lengths)

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        if model_provider not in ["OpenAI", "Anthropic", "zhipu", "HF-chat", "HF-instruct"]:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
        if model_provider == "Anthropic" and "claude" not in self.model_name:
            raise ValueError("If the model provider is 'Anthropic', the model name must include 'claude'. See https://docs.anthropic.com/claude/reference/selecting-a-model for more details on Anthropic models")
        
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        

        if not self.openai_api_key and not os.getenv('OPENAI_API_KEY'):
            raise ValueError("Either openai_api_key must be supplied with init, or OPENAI_API_KEY must be in env. Used for evaluation model")
        else:
            self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        self.zhipu_api_key = zhipu_api_key or os.getenv('ZHIPU_API_KEY')

        if self.model_provider == "Anthropic":
            if not self.anthropic_api_key and not os.getenv('ANTHROPIC_API_KEY'):
                raise ValueError("Either anthropic_api_key must be supplied with init, or ANTHROPIC_API_KEY must be in env.")
            else:
                self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
            
        if not self.model_name:
            raise ValueError("model_name must be provided.")
        
        self.model_to_test_description = self.model_name

        self.RANDOM_NEEDLE_CITIES = ['Berlin', 'Thimphu', 'Seattle', 'Mexico City', 'Chicago', 'Lagos', 
                             'Yerevan', 'Johannesburg', 'Almaty', 'Colombo', 'Damascus', 'Athens', 
                             'Dakar', 'Bangkok', 'Doha', 'Moscow', 'Kigali', 'Bratislava', 'Jakarta', 
                             'Tashkent', 'Amman', 'Tokyo', 'Toronto', 'Victoria', 'Budapest', 'Vienna', 
                             'Sydney', 'Sofia', 'Maputo', 'Port Louis', 'Lima', 'Los Angeles', 'Oslo', 
                             'Buenos Aires', 'Sarajevo', 'Delhi', 'Kampala', 'Vancouver', 'Antananarivo', 
                             'Istanbul', 'Beirut', 'Shanghai', 'Khartoum', 'Bangalore', 'Dubai', 'Paris', 
                             'Tunis', 'Lisbon', 'Mumbai', 'Amsterdam', 'Copenhagen', 'Madrid', 'San Francisco', 
                             'Cairo', 'Melbourne', 'Bucharest', 'Kuala Lumpur', 'Baghdad', 'Brussels', 'Seoul', 
                             'Belgrade', 'Astana', 'Rabat', 'Barcelona', 'Helsinki', 'Yangon', 'Ho Chi Minh City', 'Santiago', 'Nairobi']
        self.RANDOM_NEEDLE_CITIES = list(set(self.RANDOM_NEEDLE_CITIES)) 
