import xml.etree.ElementTree as ET
import xmltodict
import json
import pandas as pd
#import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde,kendalltau
from sklearn.linear_model import LinearRegression,Ridge,LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix,hstack
from sklearn.metrics import roc_auc_score

category = 'physics'#'physics'

reload_data = False
if reload_data:
	name = category + '/Posts.xml'

	tree = ET.parse(name)
	xml_data = tree.getroot()
	#here you can change the encoding type to be able to set it to the one you need
	xmlstr = ET.tostring(xml_data, encoding='utf-8', method='xml')

	data_dict = dict(xmltodict.parse(xmlstr))

	df = pd.DataFrame(data_dict['posts']['row'])


	print(df.shape)
	print(df.head())

	df.to_csv(category + '.csv')
else:
	print('Loading dataframe...')
	df = pd.read_csv(category + '.csv')

mask = (~df['@Title'].isna())&(df['@ClosedDate'].isna())
titles = df['@Title'][mask].values
bodies = df['@Body'][mask].values
scores = df['@Score'][mask].values
views = df['@ViewCount'][mask].values
#answered = ~(df['@AcceptedAnswerId'][mask].isna())
answered = df['@AnswerCount'][mask]>1
corr_score = np.log10(np.maximum(scores,1))

print(f'Answered: {np.mean(answered)}')
print(titles.shape)
#print(titles[:10])

with open('custom_stop_words.txt','r') as f:
	stop_words = f.read().splitlines()
print(stop_words)

#Counts non-stop-words in the titles.
print('Building one-hot encoding...')
vectorizer = CountVectorizer(min_df=100,vocabulary=stop_words)
X = vectorizer.fit_transform(titles)
word_counts = np.array(X.sum(axis=0))[0]
idx = np.argsort(-word_counts)
vocab = np.array(vectorizer.get_feature_names())
# for i,(word,count) in enumerate(zip(vocab[idx],word_counts[idx])):
# 	#if count>100:
# 	print(f'{word}: {count}')
# exit()
print(X.shape)
fit_score = True
if fit_score:
	print('Fitting score...')
	#y = np.log(views+1)
	y=corr_score

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

	#Fit a model
	model = Ridge(alpha=100)
	model.fit(X_train,y_train)#,sample_weight=np.exp(y_train))
	y_pred = model.predict(X_train)
	tau,pvalue = kendalltau(y_train,y_pred)
	r2 = model.score(X_train,y_train)#,sample_weight=np.exp(y_train))
	print(f'Train R^2: {r2}')
	print(f'Train Tau: {tau}')

	y_pred = model.predict(X_test)
	tau,pvalue = kendalltau(y_test,y_pred)
	r2 = model.score(X_test,y_test)#,sample_weight=np.exp(y_test))
	print(f'Test R^2: {r2}')
	print(f'Test Tau: {tau}')

	idx = np.argsort(model.coef_)
	#idx = np.array(idx,dtype=int)
	extreme_idx = np.concatenate([idx[:10],idx[-10:]])
	for w,c in zip(vocab[extreme_idx],model.coef_[extreme_idx]):
		print(f'{w}: {c}')

fit_answered = True
if fit_answered:
	print('Fitting answered...')
	#y = np.log(views+1)
	y=answered

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

	#Fit a model
	model = LogisticRegression(max_iter=1000)
	model.fit(X_train,y_train)#,sample_weight=np.exp(y_train))
	y_pred = model.predict(X_train)
	r2 = model.score(X_train,y_train)#,sample_weight=np.exp(y_train))
	ra = roc_auc_score(y_train,y_pred)-0.5
	print(f'Train R^2: {r2}')
	print(f'Train RA: {ra}')

	y_pred = model.predict(X_test)
	r2 = model.score(X_test,y_test)#,sample_weight=np.exp(y_test))
	ra = roc_auc_score(y_test,y_pred)-0.5
	print(f'Test R^2: {r2}')
	print(f'Test RA: {ra}')

	idx = np.argsort(model.coef_[0])
	#idx = np.array(idx,dtype=int)
	extreme_idx = np.concatenate([idx[:20],idx[-20:]])
	for w,c in zip(vocab[extreme_idx],model.coef_[0][extreme_idx]):
		print(f'{w}: {c}')