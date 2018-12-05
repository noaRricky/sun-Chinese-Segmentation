
from flask import Flask, request, jsonify
from flask_cors import CORS

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

import my_library

# load web application
app = Flask(__name__)
CORS(app, supports_credentials=True)

# load machine predictor
archive = load_archive('store/exp_crf/model.tar.gz')
predictor = Predictor.from_archive(archive, 'sentence-segment')


@app.route('/')
def index():
    return 'Hello World'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sentence = request.json['sentence']
        result = predictor.predict(sentence=sentence)
        result = jsonify(result)
        return result


if __name__ == '__main__':
    app.run(debug=True)