from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model_judi_nb.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    title = data['title']
    prediction = model.predict([title])[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
