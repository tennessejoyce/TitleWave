from ktrain import load_predictor

#Predict title quality
predictor = load_predictor('../../ktrain_predictor_overflow')

def predict_quality(title):
	prob = predictor.predict_proba([title])[0,1]
	#Format as a string
	return f'{100*prob:.1f}%'


#To understand which words are contributing to the prediction.
#predictor.explain()
