from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name, static_url_path='/static')

# Load the model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def show_form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data from the request
        itching = int(request.form['itching'])
        skin_rash = int(request.form['skin_rash'])
        nodal_skin_eruptions = int(request.form['nodal_skin_eruptions'])
        continuous_sneezing = int(request.form['continuous_sneezing'])
        shivering = int(request.form['shivering'])
        # Add the other fields in a similar manner
        
        # Process the data and make the prediction using the loaded model
        data = {
            "itching": itching,
            "skin_rash": skin_rash,
            "nodal_skin_eruptions": nodal_skin_eruptions,
            "continuous_sneezing": continuous_sneezing,
            "shivering": shivering,
            # Add the other fields in a similar manner
        }

        # Convert the dictionary to a DataFrame and reshape it for prediction
        df = pd.DataFrame([data])
        first_element = df.iloc[0]
        first_element_array = np.array(first_element)
        first_element_reshaped = first_element_array.reshape(1, -1)

        # Make the prediction
        prediction = model.predict(first_element_reshaped)
        prediction_value = "Safe" if prediction == 0 else "Not Safe"

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction_value})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
