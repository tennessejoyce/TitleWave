import torch
from torch.utils.data import DataLoader
from data_loader import PostDataset
from training_loops import fit
from transformers import BertForSequenceClassification
from torch.optim import AdamW, Adam
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch.optim.lr_scheduler import ExponentialLR
from pymongo import MongoClient
from datetime import datetime
import numpy as np

if __name__=='__main__':

	# Hyperparameters
	batch_size = 64
	max_epochs = 20
	train_steps_per_epoch = 8*1024//batch_size
	val_steps_per_epoch = train_steps_per_epoch

	# Specify training set
	start_date = datetime(2019, 1, 1)
	end_date = datetime(2020, 1, 1)
	doc_filter = {'CreationDate': {'$gte': start_date, '$lt': end_date}}

	database='titlewave'
	forum_name='physics'

	# Load the data
	print('Loading dataset...')
	collection = MongoClient()[database][f'{forum_name}_posts']
	data_set = PostDataset(collection, doc_filter=doc_filter)
	data_loader = data_set.batch_generator(batch_size=batch_size)

	print(f'{len(data_set)} documents found...')

	# Load the model
	print('Loading model...')
	model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 1)
	model.cuda()

	# Freeze all layers except the last at first.
	for p in model.parameters():
		p.requires_grad = False
	for p in model.classifier.parameters():
		p.requires_grad = True

	# Specify the optimizer and loss function
	print('Setting hyperparameters...')
	loss_function = L1Loss() #BCEWithLogitsLoss()
	optimizer = Adam(model.parameters(), lr=1)
	scheduler = ExponentialLR(optimizer, gamma=0.9)

	# Train the model
	print('Training model...')
	fit(model, data_loader, optimizer, loss_function, scheduler,
		train_steps_per_epoch, val_steps_per_epoch, max_epochs)

