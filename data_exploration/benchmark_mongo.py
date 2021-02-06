from pymongo import MongoClient
from time import time
from datetime import datetime
import numpy as np

database='titlewave'
forum_name='physics'

# Specify training set
start_date = datetime(2019, 1, 1)
end_date = datetime(2020, 1, 1)
doc_filter = {'CreationDate': {'$gte': start_date, '$lt': end_date}}

collection = MongoClient()[database][f'{forum_name}_posts']

# How long to parse the database with different approaches?
cursor_length = collection.count_documents(doc_filter)
cursor = collection.find(doc_filter)

#Ordered pass
start = time()
for doc in cursor:
	pass
print(f'Ordered pass: {time()-start:.4f} s')

#Random order pass
def random_order(size):
	out = np.arange(size)
	np.random.shuffle(out)
	return out



# Rebuild the cursor
cursor_length = collection.count_documents(doc_filter)
cursor = collection.find(doc_filter)

start = time()
for i in random_order(cursor_length):
	cursor[int(i)]
print(f'Ordered pass: {time()-start:.4f} s')