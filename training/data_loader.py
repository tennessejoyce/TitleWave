import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from time import time


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_batched_indices(size, batch_size, shuffle=True):
	# Create an array of the possible indices
	out = np.arange(size)
	if shuffle:
		# Shuffle it in place
		np.random.shuffle(out)
	# Drop elements from the end so that the array is evenly divided into batches
	remainder = size % batch_size
	if remainder>0:
		out = out[:-remainder]
	# Divide into batches
	out = np.reshape(out, (-1, batch_size))
	return out


class PostDataset (Dataset):
	def __init__(self, collection, doc_filter={}):
		cursor = collection.find(doc_filter)
		titles = []
		self.labels = []
		for doc in cursor:
			titles.append(doc['Title'])
			self.labels.append(float(doc['ViewCount']))
		self.labels = torch.from_numpy(np.array(self.labels))
		tokenized = tokenizer(titles, padding=True, truncation=True, return_tensors='pt')
		self.input_ids, self.attention_mask = tokenized.input_ids, tokenized.attention_mask

	def __len__(self):
		return self.labels.shape[0]

	def batch_generator(self, batch_size, shuffle=True):
		#Drop the last enties to make an even number of batches.
		batched_indices = get_batched_indices(len(self), batch_size, shuffle)
		for idx in batched_indices:
			yield self.input_ids[idx].cuda(), self.attention_mask[idx].cuda(), self.labels[idx].cuda()


