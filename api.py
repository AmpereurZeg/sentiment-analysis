from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.datasets import imdb
import numpy as np
from tensorflow.keras.preprocessing import sequence

app = Flask(__name__)

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
	numericText = np.array([word_index[word] if (word in word_index) and (word_index[word]<2500) else 0 for word in tokenizedText])
	numeric_inp = sequence.pad_sequences([numericText],maxlen=70)
	num_predict = loaded_model.predict(numeric_inp)
	for item in num_predict[0]:
		if item>=0.5:
			gru_prediction='positif'
		else:
		 	gru_prediction='negatif'

	return gru_prediction

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/join', methods=['GET','POST'])
def my_form_post():
	text1 = request.form['text1']
	model=load_model()
	gru_prediction = "This opinion is " +predict(text1,model)
	result = {
        "output": gru_prediction
    }
	result = {str(key): value for key, value in result.items()}
	return jsonify(result=result)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)