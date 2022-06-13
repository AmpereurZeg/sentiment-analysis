#predicting for new datasets
#predicting for new datasets
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.datasets import imdb
import numpy as np
from tensorflow.keras.preprocessing import sequence


def load_model():
	json_file = open('model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	#load weights into new model
	loaded_model.load_weights("model.h5")

	#compile and evaluate loaded model
	loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

	tokenizer = Tokenizer(num_words=2500, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=' ')
	return loaded_model

def predict(text,loaded_model):
	word_index = imdb.get_word_index()
	tokenizedText= text_to_word_sequence(text)
	numericText = np.array([word_index[word] if (word in word_index) and (word_index[word]<25000) else 0 for word in tokenizedText])
	numeric_inp = sequence.pad_sequences([numericText],maxlen=100)
	num_predict = loaded_model.predict(numeric_inp)
	for item in num_predict[0]:
		if item>=0.5:
			gru_prediction='positif'
		else :
			gru_prediction='negatif'
	return gru_prediction

print("-----------------------------------------------")

if __name__ == '__main__':
	model=load_model()
	print()
	print()
	print()
	print()
	print()
	print()
	print("------------------------------------------")
	print()
	print()
	print()
	print("**************************************Sentiment Analysis using GRU algorithme*******************************************")
	print()
	print()
	print()
	while(1):
		print("enter your text please...")
		print()
		print(">>>",end='')
		text=input()
		gru_prediction=predict(text,model)
		print()
		print()
		print("this opinion is "+gru_prediction)

		print()
		print()
