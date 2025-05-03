from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model dan vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    title = data.get('title', '')
    vectorized = vectorizer.transform([title])
    prediction = model.predict(vectorized)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run()
