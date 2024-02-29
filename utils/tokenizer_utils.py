
import tiktoken
from anthropic import AsyncAnthropic, Anthropic
from openai import AsyncOpenAI
from transformers import AutoTokenizer
import torch
class TokenizerHandler:
    def __init__(self, model_name, model_provider):
        self.model_name = model_name
        self.model_provider = model_provider
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print('self.model_provider',self.model_provider)

        if self.model_provider == "OpenAI":
            try:
                self.enc = tiktoken.encoding_for_model(self.model_name)
            except:
                self.enc = tiktoken.encoding_for_model("gpt-4-turbo-preview")
        elif self.model_provider == "zhipu":
            self.enc = tiktoken.encoding_for_model("gpt-4-turbo-preview")
        elif self.model_provider == "Anthropic":
            self.enc = Anthropic().get_tokenizer()
        elif "HF" in self.model_provider:
            self.enc = AutoTokenizer.from_pretrained(self.model_name,use_fast=False)
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic' or 'HF' ")

    def estimate_token_length(self, text):
        tokens = self.encode_text_to_tokens(text)    
        token_length = len(tokens)

        return token_length


    def encode_text_to_tokens(self, text):
        if self.model_provider == "OpenAI" and "gpt" in self.model_name.lower():
            return self.enc.encode(text)
        elif self.model_provider == "OpenAI" in self.model_name.lower():
            return self.enc.encode(text)
        elif self.model_provider == "zhipu":
            return self.enc.encode(text)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        elif "HF" in self.model_provider:
            return self.enc.encode(text,add_special_tokens = False)
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic' or 'HF")
    

    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider == "OpenAI" and "gpt" in self.model_name.lower():
            return self.enc.decode(tokens[:context_length])
        elif self.model_provider == "zhipu":
            return self.enc.decode(tokens[:context_length])
        elif self.model_provider == "OpenAI" in self.model_name.lower():
            return self.enc.decode(tokens[:context_length])
        elif self.model_provider == "Anthropic":
            # Assuming you have a different decoder for Anthropic
            return self.enc.decode(tokens[:context_length])
        elif "HF" in self.model_provider:
            return self.enc.decode(tokens[:context_length])
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic' or 'HF'")
    def encode_and_trim(self, context, context_length):
        tokens = self.encode_text_to_tokens(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context

def test(model_name, model_provider):
    tokenizer_handler = TokenizerHandler(model_name, model_provider)
    print('input:',text)
    # Test estimate_token_length
    token_length = tokenizer_handler.estimate_token_length(text)
    print('token_length:',token_length)
    assert isinstance(token_length, int), "Token length should be an integer"

    # Test encode_text_to_tokens
    tokens = tokenizer_handler.encode_text_to_tokens(text)
    print('tokens',tokens)
    assert isinstance(tokens, list), "Tokens should be a list"

    # Test decode_tokens
    decoded_text = tokenizer_handler.decode_tokens(tokens)
    print('decoded_text',decoded_text)
    assert isinstance(decoded_text, str), "Decoded text should be a string"

    trim_text = tokenizer_handler.encode_and_trim(text,5)
    print('trim:',trim_text)



# Example usage
if __name__ == "__main__":

    text = "This is a sample text to estimate its token length."
    print('-------------testing OpenAI-----------')
    model_name = "gpt-4-turbo-preview" # You can replace this with any model name from Hugging Face
    model_provider = "OpenAI" # Replace with "Anthropic" if using Anthropic
    test(model_name, model_provider)

    print('-------------testing Anthropic-----------')
    model_name = "claude-2" # You can replace this with any model name from Hugging Face
    model_provider = "Anthropic" # Replace with "Anthropic" if using Anthropic
    test(model_name, model_provider)

    print('-------------testing HF-----------')
    model_name = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO" # You can replace this with any model name from Hugging Face
    model_provider = "HF" # Replace with "Anthropic" if using Anthropic
    test(model_name, model_provider)
    