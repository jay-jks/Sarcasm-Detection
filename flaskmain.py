from flask import Flask, render_template, request
import pickle

# Define the Flask app
app = Flask(__name__)

# Load the saved model
loaded_model = pickle.load(open('model.pkl', 'rb'))
text_transformer=pickle.load(open('modell.pkl', 'rb'))

def requestResults(result):
    if result == 0:
        return "Not-Sarcastic"
    else:
        return "Sarcastic"

# Use the loaded model to make predictions    
def sardet(text): 
    input_data = [text]
    prediction = loaded_model.predict(text_transformer.transform(input_data))
    return prediction

# Define the route for the web app's homepage
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    pred = sardet(message)
    result=requestResults(pred)
    return render_template('index.html', prediction_text=result)


# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)