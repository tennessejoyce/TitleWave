import pandas as pd
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer
import torch
import os
from glob import glob
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from tqdm import tqdm

#Train on GPU if available (go to 'Edit/Notebook Settings' in Colab to enable)
if torch.cuda.is_available():
  device = torch.device("cuda")
  print('Running on GPU...')
else:
  print('GPU not found...')
  device = torch.device("cpu")
  print('Running on CPU...')


num_steps = 10
train_per_step = 100
test_per_step = 4




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



def train_data(directory):
  #A generator to read the data out of multiple csv files.
  files = sorted(glob(directory+"/*"))
  for f in files:
    #Load the example data
    df = pd.read_csv('example.csv',index_col=0)
    X,y= df['Body'],df['Title']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=42)
    for step in range(num_steps):
      for i,(X_i,y_i) in enumerate(zip(X_train,y_train)):
        if i>=train_per_step:
          break
        yield 'train',i,X_i,y_i
      for i,(X_i,y_i) in enumerate(zip(X_test,y_test)):
        if i>=test_per_step:
          break
        yield 'test',i,X_i,y_i
      yield 'checkpoint',step,X_i,y_i



#Fine-tuning T5 to summarize a question into a title.
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=0.001/128)

#Evaluation mode disables dropout layers, which aren't needed when we only train for a single epoch.
model.eval()

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
rouge1 = 0
rouge2 = 0
summaries = []

#Training loop
for mode,i,X_i,y_i in tqdm(train_data('training_set'),total=train_per_step+test_per_step):
  if mode=='train':
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
  elif mode=='test':
    #Compute the loss, ROUGE score, and some example titles.
    #Print some example titles, to gauge how well it's doing (needs many more training examples to produce reasonable results)
    with torch.no_grad():
      X_i = tokenizer.encode(X_i, return_tensors="pt",max_length=512).to(device)
      summary = model.generate(X_i,max_length=200,num_beams=4,no_repeat_ngram_size=3)[0]
      summary = tokenizer.decode(summary, skip_special_tokens=True)
      scores = scorer.score(y_i,summary)
      rouge1 += scores['rouge1'].fmeasure
      rouge2 += scores['rouge2'].fmeasure
  elif mode=='checkpoint':
    #Save the model
    print('Saving model...')
    checkpoint_directory = f't5-6-9-{i}'
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)
    model.save_pretrained(checkpoint_directory)
    #Save the validation scores.
    with open(checkpoint_directory+'/validation.txt','w') as f:
      f.write(rouge1/test_per_step)
      f.write(rouge2/test_per_step)
      f.writelines(summaries)
    rouge1 = 0
    rouge2 = 0
#Load back with T5ForConditionalGeneration.from_pretrained('t5-finetuned')