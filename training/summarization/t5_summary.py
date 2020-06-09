import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from tqdm.notebook import tqdm
#Train on GPU if available (go to 'Edit/Notebook Settings' in Colab to enable)
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

#Load the example data
df = pd.read_csv('example.csv',index_col=0)
print(list(df))
#Fine-tuning T5 to summarize a question into a title.
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=3e-5)

X,y= df['Body'],df['Title']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, random_state=42)

#Training loop
for epoch in range(2):
  model.train()
  for X_i,y_i in tqdm(zip(X_train,y_train),total=len(X_train)):
    #Encode the titles (the prefix 'summarize: ' tells it to summarize)
    X_i = tokenizer.encode('summarize: '+X_i, return_tensors="pt",max_length=512,pad_to_max_length=True).to(device)
    y_i = tokenizer.encode(y_i, return_tensors="pt",max_length=512,pad_to_max_length=True).to(device)
    #Foward pass through the network
    loss = model(input_ids=X_i, lm_labels=y_i)[0]
    #Backward pass to compute the gradients
    loss.backward()
    #Update the parameters with gradient descent
    optimizer.step()
    optimizer.zero_grad()
    model.zero_grad()

  #Switch to evaluation mode
  model.eval()

  #Print some example titles, to gauge how well it's doing (needs many more training examples to produce reasonable results)
  for X_i,y_i in zip(X_test,y_test):
    X_i = tokenizer.encode(X_i, return_tensors="pt",max_length=512).to(device)
    summary = model.generate(X_i,max_length=200,num_beams=4,no_repeat_ngram_size=3)[0]
    summary = tokenizer.decode(summary, skip_special_tokens=True)
    print(summary)

#Save the model
print('Saving model...')
directory = 't5-finetuned'
if not os.path.exists(directory):
    os.makedirs(directory)
model.save_pretrained(directory)
#Load back with T5ForConditionalGeneration.from_pretrained('t5-finetuned')