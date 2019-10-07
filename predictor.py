import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

def get_model():
    global model
    model = load_model('static/models/model.h5')

def con2vec(input):
    with open('static/models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        tokenized_text = tokenizer.texts_to_sequences([input])
        vector = pad_sequences(tokenized_text, maxlen=100)
        return vector
    
def make_prediction(input):
    vector = con2vec(input)
    prediction = model.predict([vector], batch_size=1, verbose=0)
    return prediction

def predict(input):
    if len(input) > 500:
        input = input[:500]
    predictions = make_prediction(input)
    return (input,predictions)

if __name__ == '__main__':
    get_model()
    input,predictions = predict("This is so great")
    print(input)
    print(predictions)
    input,predictions = predict("This is so bad")
    print(input)
    print(predictions)
    input,predictions = predict("This is so insulting")
    print(input)
    print(predictions)