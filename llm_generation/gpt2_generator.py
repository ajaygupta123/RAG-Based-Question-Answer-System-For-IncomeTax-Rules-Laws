# from transformers import DistilGPT2LMHeadModel, GPT2Tokenizer

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Generation Class using DistilGPT-2
class DistilGPT2Generation:
    def __init__(self, model_name='distilgpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate(self, context, query):
        input_text = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        inputs = self.tokenizer.encode(input_text, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=150, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

