import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def title_transform(title):
	return tokenizer.encode(title, padding='max_length', truncation=True, return_tensors='pt')

class PostDataset (Dataset):
	def __init__(self, collection, doc_filter={}, title_transform=title_transform):
		self.cursor_length = collection.count_documents(doc_filter)
		self.cursor = collection.find(doc_filter)
		self.title_transform = title_transform


	def __len__(self):
		return self.cursor_length

	def __getitem__(self, i):
		doc = self.cursor[i]
		X = title_transform(doc['Title'])
		y = doc['AnswerCount'] > 0
		return X, y
