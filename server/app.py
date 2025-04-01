from flask import Flask, request, jsonify
from inference import load_model_and_components, predict_sentiment
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and components
model, vectorizer, label_encoder, device = load_model_and_components()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    comment = data.get('comment', '')

    if not comment:
        return jsonify({'error': 'No comment provided'}), 400

    result = predict_sentiment(comment, model, vectorizer, label_encoder, device)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
