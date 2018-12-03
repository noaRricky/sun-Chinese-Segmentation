
from flask import Flask, request, jsonify
from flask_cors import CORS

from allennlp.predictors import Predictor

# load web application
app = Flask(__name__)
CORS(app, supports_credentials=True)

# load machine comprehension predictor
# mc_predictor = Predictor.from_path(
#     './model/bidaf-model-2017.09.15-charpad.tar.gz')


@app.route('/')
def index():
    print("This is the index page")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        passage = request.json['long_text_input']
        question = request.json['short_text_input']
        result = mc_predictor.predict(passage=passage, question=question)
        result = jsonify(result)
        return result


if __name__ == '__main__':
    app.run(debug=True)