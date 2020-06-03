import pandas as pd
import numpy as np
import ktrain
from ktrain import text
from sklearn.model_selection import train_test_split

#Titles are stored in this csv file.
#I'm reading them out into a list of strings
filename = 'physics.csv'
df = pd.read_csv(directory+'/physics.csv')

#Select only the entries that are questions (not answers) and weren't closed.
mask = (~df['@Title'].isna())&(df['@ClosedDate'].isna())
titles = df['@Title'][mask].values
answered = df['@AnswerCount'][mask].values>=1

#Split into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(titles, answered, test_size=0.2, random_state=42)

#Initialize the BERT model.
class_names = ['unanswered','answered']
MODEL_NAME = 'distilbert-base-uncased'
t = text.Transformer(MODEL_NAME, maxlen=500, class_names=class_names)
trn = t.preprocess_train(x_train, y_train)
val = t.preprocess_test(x_test, y_test)
model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)

#Fine tune the model. After 2 epochs, it seems to overfit for me, but that probably depends on the task.
learner.fit_onecycle(3e-5, 2)

#Get a separate object to make predictions.
predictor = ktrain.get_predictor(learner.model, preproc=t)
#Save it to a file so we can use it elsewhere to make predictions.
predictor.save('ktrain_predictor')

#Predict the probabilities of getting an answer on the test set.
y_pred = predictor.predict_proba(x_test)[:,1]

#Compute the ROC-AUC score (a good metric for imbalanced classification).
ra = roc_auc_score(y_test,y_pred)
print(f'Test ROC-AUC: {ra}')

#Separate the titles into quantiles (1 to 5 stars) based on the predicted probabilities,
#then compute the actual probability of an answer within each quantile.
divisions = [np.quantile(y_pred,q) for q in [0,0.2,0.4,0.6,0.8,1]]
for i in range(5):
  qmask = (y_pred > divisions[i])&(y_pred <= divisions[i+1])
  print(f'{i+1} stars: {np.mean(y_test[qmask]):.3f}   {np.sum(qmask)}')


#To understand which words are contributing to the prediction.
#predictor.explain()
