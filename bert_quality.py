import pandas as pd
import numpy as np
import ktrain
from ktrain import text
from sklearn.model_selection import train_test_split
filename = 'physics.csv'
df = pd.read_csv(directory+'/physics.csv')

question_mask = (~df['@Title'].isna())&(df['@ClosedDate'].isna())
titles = df['@Title'][question_mask].values
#bodies = df['@Body'][question_mask].values


def preprocess(s):
	#Remove certain special characters, replace with spaces.
	remove_list = ['\n','<p>','</p>']
	for r in remove_list:
		s = s.replace(r,' ')
	return s

mask = (~df['@Title'].isna())&(df['@ClosedDate'].isna())
titles = df['@Title'][mask].values
#bodies = df['@Body'][mask].map(preprocess).values
#scores = df['@Score'][mask].values
#corr_score = np.log10(np.maximum(scores,1))
#views = df['@ViewCount'][mask].values
#answered = ~(df['@AcceptedAnswerId'][mask].isna())
answered = df['@AnswerCount'][mask].values>=1

#Take only the most recent
titles = titles
answered = answered

x_train, x_test, y_train, y_test = train_test_split(titles, answered, test_size=0.2, random_state=41)

			

class_names = ['unanswered','answered']
MODEL_NAME = 'distilbert-base-uncased'
t = text.Transformer(MODEL_NAME, maxlen=500, class_names=class_names)
trn = t.preprocess_train(x_train, y_train)
val = t.preprocess_test(x_test, y_test)
model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)

#Fine tune the model. After 2 epochs, it seems to overfit.
learner.fit_onecycle(3e-5, 2)

predictor = ktrain.get_predictor(learner.model, preproc=t)
predictor.save('ktrain_predictor')

y_pred = predictor.predict_proba(x_test)[:,1]

ra = roc_auc_score(y_test,y_pred)
print(f'Test ROC-AUC: {ra}')

divisions = [np.quantile(y_pred,q) for q in [0,0.2,0.4,0.6,0.8,1]]
for i in range(5):
  qmask = (y_pred > divisions[i])&(y_pred <= divisions[i+1])
  print(f'{i+1} stars: {np.mean(y_test[qmask]):.3f}   {np.sum(qmask)}')