import torch
from torch.utils.data import DataLoader
from data_loader import PostDataset
from training_loops import fit
from transformers import BertForSequenceClassification
from torch.optim import AdamW, Adam
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ExponentialLR
from pymongo import MongoClient
from datetime import datetime

if __name__=='__main__':

	# Hyperparameters
	batch_size = 8
	max_epochs = 10
	train_steps_per_epoch = 100
	val_steps_per_epoch = 10

	# Specify training set
	start_date = datetime(2014, 1, 1)
	end_date = datetime(2015, 1, 1)
	doc_filter = {'CreationDate': {'$gte': start_date, '$lt': end_date}}

	database='titlewave'
	forum_name='physics'

	# Load the data
	print('Loading dataset...')
	collection = MongoClient()[database][f'{forum_name}_posts']
	data_set = PostDataset(collection, doc_filter=doc_filter)

	data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)


	# Load the model
	print('Loading model...')
	model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 3)

	# Specify the optimizer and loss function
	print('Setting hyperparameters...')
	loss_function = BCEWithLogitsLoss()
	optimizer = Adam(model.parameters(), lr=1e-3)
	scheduler = ExponentialLR(optimizer, gamma=0.9)

	# Train the model
	fit(model, data_loader, optimizer, loss_function, scheduler,
		train_steps_per_epoch, val_steps_per_epoch, max_epochs)


