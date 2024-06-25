from sklearn.metrics import precision_score, recall_score, f1_score
from rank_bm25 import BM25Okapi
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Example data
queries = ["What is the income tax rate for 2021?"]
documents = ["The income tax rate for 2021 is 10%.", "For 2021, the income tax rate is 10%."]
relevant_docs = [1]  # Assuming the second document is relevant

# Simulate the retrieval process
bm25 = BM25Okapi([doc.split() for doc in documents])
scores = bm25.get_scores(queries[0].split())
top_k_indices = scores.argsort()[-1:][::-1]  # Top-1 document index

# Precision, Recall, and F1 Score
precision = precision_score(relevant_docs, top_k_indices, average='micro')
recall = recall_score(relevant_docs, top_k_indices, average='micro')
f1 = f1_score(relevant_docs, top_k_indices, average='micro')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Generate response using GPT-2
input_ids = tokenizer.encode(queries[0], return_tensors='pt')
outputs = model.generate(input_ids, max_length=50)
generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# BLEU Score
bleu_score = sentence_bleu([documents[1].split()], generated_response.split())
print(f"BLEU Score: {bleu_score}")

# ROUGE Score
rouge = Rouge()
rouge_scores = rouge.get_scores(generated_response, documents[1])
print(f"ROUGE Scores: {rouge_scores}")
