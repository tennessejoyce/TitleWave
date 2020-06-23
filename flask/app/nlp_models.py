from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer,BertForSequenceClassification
from nltk.tokenize import sent_tokenize
import torch
import re
import numpy as np

bert_location = '../../models/BERT'
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained(bert_location, num_labels = 3)
bert_model.eval()

t5_location = '../../models/T5'
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained(t5_location)
t5_model.eval()

def clean_v3(text):
  #Remove code blocks, urls, and html tags.
  text = re.sub(r'<code[^>]*>(.+?)</code\s*>', '', text,flags=re.DOTALL | re.MULTILINE)
  text = re.sub(r'<div[^>]*>(.+?)</div\s*>', '', text,flags=re.DOTALL | re.MULTILINE)
  text = re.sub(r'<blockquote[^>]*>(.+?)</blockquote\s*>', '', text,flags=re.DOTALL | re.MULTILINE)
  text = re.sub('<.*?>', '', text)
  text = text.replace('&quot;','"')
  text = re.sub(r'http\S+', '', text)
  text = re.sub(r'www.\S+', '', text)
  return text

def suggest_title(body):
	with torch.no_grad():
		body = 'summarize: ' + clean_v3(body)
		body = t5_tokenizer.encode(body, return_tensors="pt",max_length=512)
		summary = t5_model.generate(body,max_length=100,num_beams=16,no_repeat_ngram_size=2)[0]
		summary = t5_tokenizer.decode(summary, skip_special_tokens=True)
		#Take only the first sentence.
		summary = sent_tokenize(summary)[0]
	return summary

def evaluate_title(title,metric=0):
	out = []
	with torch.no_grad():
		title = bert_tokenizer.encode(title, return_tensors="pt")
		logit = bert_model(title)[0][0][metric].cpu().numpy()
		prob = np.exp(logit)/(1+np.exp(logit))
	return f'{100*prob:.1f}%'
