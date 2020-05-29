import xml.etree.ElementTree as ET
import xmltodict
import json
import pandas as pd
#import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde,kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix,hstack

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

question_mask = ~df['@Title'].isna()
titles = df['@Title'][question_mask].values
bodies = df['@Body'][question_mask].values
scores = df['@Score'][question_mask].values
corr_score = np.log10(np.maximum(scores,1))

print(titles.shape)
print(titles[:10])

with open(category+'_titles.txt','w', encoding='utf-8') as f:
	f.writelines([t+'\n' for t in titles])
exit()


count_frequencies = False
if count_frequencies:
	#Counts non-stop-words in the titles.
	print('Building one-hot encoding...')
	vectorizer = CountVectorizer(stop_words='english')
	X = vectorizer.fit_transform(titles)
	word_counts = np.array(X.sum(axis=0))[0]
	idx = np.argsort(-word_counts)
	vocab = np.array(vectorizer.get_feature_names())
	for i,(word,count) in enumerate(zip(vocab[idx],word_counts[idx])):
		if i>20:
			exit()
		print(f'{word}: {count}')




#Build title features

question_mark = np.array([t[-1]=='?' for t in titles],dtype=int)
print('Analyze question mark...')
print(np.mean(question_mark))
print(np.mean(corr_score[question_mark]))
print(np.mean(corr_score[1-question_mark]))
#exit()



#Compute the mean TF-IDF scores of each title.
print('Building TF-IDF encoding...')
vectorizer = CountVectorizer(stop_words='english',min_df=1000)
tfidf = vectorizer.fit_transform(bodies)
body_length = np.array((tfidf>0).sum(axis=1))[:,0]+1
body_sum_tfidf = np.array(tfidf.sum(axis=1))[:,0]



#Build title features.
print('Building title features...')
title_tfidf = vectorizer.transform(titles)
length = np.array((title_tfidf>0).sum(axis=1))[:,0]+1
sum_tfidf = np.array(title_tfidf.sum(axis=1))[:,0]
mean_tfidf = sum_tfidf/length
similarity = np.array(tfidf.multiply(title_tfidf).sum(axis=1))[:,0]
mean_similarity = similarity/(1+body_sum_tfidf)/(1+sum_tfidf)

title_features = [csr_matrix(m[:,None]) for m in [body_length,question_mark,length,mean_tfidf,sum_tfidf,similarity,mean_similarity]]
#question_mark,length,mean_tfidf,sum_tfidf,similarity,mean_similarity,


print('Predicting...')





#Makes a scatter plot where each point is colored according to the local point density.
def density_plot(x,y):
	#Use kernel density estimation to color the points.
	points = np.vstack([x,y])
	density = gaussian_kde(points)(points)
	#Sort the points so that the highest densities appear on top. This looks a bit better.
	idx = density.argsort()
	x,y,density = x[idx], y[idx], density[idx]
	#Draw the scatter plot.
	plt.scatter(x, y, c=density, s=5)


#density_plot(length,corr_score)
#plt.scatter()
#plt.show()

print(question_mark.shape)
print(tfidf.shape)
X = hstack(title_features+[tfidf])#np.stack([length,mean_tfidf],axis=-1) #question_mark[:,None]
y = corr_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Fit a model
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_train)
tau,pvalue = kendalltau(y_train,y_pred)
r2 = model.score(X_train,y_train)
print(f'Train R^2: {r2}')
print(f'Train Tau: {tau}')

y_pred = model.predict(X_test)
tau,pvalue = kendalltau(y_test,y_pred)
r2 = model.score(X_test,y_test)
print(f'Test R^2: {r2}')
print(f'Test Tau: {tau}')


