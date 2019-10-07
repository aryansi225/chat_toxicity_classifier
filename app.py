from flask import Flask, render_template, request
from predictor import predict, get_model

app = Flask(__name__)

get_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result',methods=["POST"])
def output():
    if request.method == 'POST':
        data = request.form['data']
        input, predictions = predict(data)
        predictions = predictions.flatten()
        return render_template('index.html', hasOutput = True, data=input, predictions=predictions)

if __name__=='__main__':
    app.run(debug=False, threaded=False)