from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
import joblib  # Use joblib for loading the model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Upload folder path

# Load the model and scaler using joblib
rfc_model = joblib.load('models/rfc_model.joblib')
scaler = joblib.load('models/scaler.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess and predict
        processed_data, df = preprocess_csv(file_path)
        predictions = rfc_model.predict(processed_data)
        
        # Create a new DataFrame for displaying results with S.No, FLOW_ID, and Prediction
        result_data = pd.DataFrame({
            'S_No': range(1, len(df) + 1),  # Serial number starting from 1
            'FLOW_ID': df['FLOW_ID'],  # FLOW_ID from the original dataframe
            'Prediction': predictions  # Directly use predictions from the model
        })
        # Convert the result DataFrame to a dictionary for rendering in the template
        result_data_dict = result_data.to_dict(orient='records')
        print(result_data_dict)

        return render_template('result.html', result=result_data_dict)
    
    return render_template('index.html')

def preprocess_csv(file_path):
    # Load CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Ensure that FLOW_ID is included in the DataFrame
    # Select the required columns for prediction, but keep FLOW_ID
    required_columns = [
        'IN_BYTES', 'ANOMALY', 'TCP_WIN_MSS_IN', 'L4_DST_PORT', 
        'TCP_WIN_MIN_IN', 'TCP_WIN_MAX_IN', 'TOTAL_FLOWS_EXP', 'OUT_BYTES', 
        'FIRST_SWITCHED', 'LAST_SWITCHED', 'FLOW_DURATION_MILLISECONDS', 
        'IN_PKTS', 'TCP_WIN_MIN_OUT', 'L4_SRC_PORT', 'TCP_FLAGS', 'FLOW_ID'
    ]
    
    # Extract the required columns from the DataFrame
    df = df[required_columns]

    # Fill missing values with 0
    df.fillna(0, inplace=True)

    # Apply the scaler (exclude 'FLOW_ID' column for scaling)
    df_scaled = scaler.transform(df.drop(columns=['FLOW_ID']))
    
    return df_scaled, df  # Return both the scaled data and original dataframe with FLOW_ID

if __name__ == '__main__':
    app.secret_key = 'your_secret_key'  # Set a secret key for sessions
    app.run(debug=True)
