import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import os
from transformers import BertTokenizer,BertForSequenceClassification,T5Tokenizer, T5ForConditionalGeneration, AdamW
from tqdm.notebook import tqdm
from scipy.special import softmax
#Train on GPU if available (go to 'Edit/Notebook Settings' in Colab to enable)
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

#Load the example data
df = pd.read_csv('example.csv',index_col=0)
print(list(df))

#Fine-tuning BERT to predict whether a question will be answered, based on the title.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=3e-5)

X,y= df['Title'],df['Answered']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=42)

#Training loop
for epoch in range(2):
  model.train()
  for X_i,y_i in tqdm(zip(X_train,y_train),total=len(X_train)):
    #Encode the titles
    X_i = tokenizer.encode(X_i, return_tensors="pt").to(device)
    y_i = torch.tensor(y_i).to(device)
    #Foward pass through the network
    loss = model(X_i, labels=y_i)[0]
    #Backward pass to compute the gradients
    loss.backward()
    #Update the parameters with gradient descent
    optimizer.step()
    optimizer.zero_grad()
    model.zero_grad()

  #Switch to evaluation mode
  model.eval()

  #Make predictions on the validation set
  probs = []
  for X_i,y_i in zip(X_test,y_test):
    X_i = tokenizer.encode(X_i, return_tensors="pt").to(device)
    y_i = torch.tensor(y_i).to(device)
    logits = model(X_i)[0][0].detach().cpu().numpy()
    probs.append(softmax(logits)[1])

  print(f'Epoch {epoch+1} ROC-AUC score: {roc_auc_score(y_test,probs):.3f}')

#Save the model
print('Saving model...')
directory = 'bert-finetuned'
if not os.path.exists(directory):
    os.makedirs(directory)
model.save_pretrained(directory)
#Load back with BertForSequenceClassification.from_pretrained('bert-finetuned')