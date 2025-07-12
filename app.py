from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    data = vectorizer.transform([message])
    prediction = model.predict(data)
    result = "💥 SPAM" if prediction[0] == 1 else "✅ NOT SPAM"
    return render_template('index.html', prediction=result, user_input=message)

if __name__ == '__main__':
    app.run(debug=True)
