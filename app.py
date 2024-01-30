from flask import Flask, render_template,redirect, session, url_for, request
from werkzeug.utils import secure_filename
import pandas as pd
from fileinput import filename
import os
#from script import detect_data                #Importing function to detect data types


UPLOAD_FOLDER = os.path.join('static','uploads')

ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'Secret key to utilize session'

@app.route('/', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
        #Upload file
        uploadedFile = request.files['file']  

        #Extracting uploaded file name
        data_filename = secure_filename(uploadedFile.filename)
        
        file_location = os.path.join((app.config['UPLOAD_FOLDER']),data_filename)
        
        uploadedFile.save(file_location)          

        session['uploaded_data_file_path'] = file_location

        if request.form['submit_button'] == 'EDA':
            return render_template('Eda.html')
        
        elif request.form['submit_button'] == 'Preprocess':
            return render_template('Preprocessing.html')
    
    else:
        return render_template('index.html')

@app.route('/preprocess')
def preprocess():
    if request.method == 'POST':
        
        data_file_path = session.get('uploaded_data_file_path',None)
        preprocess_df = pd.read_csv(data_file_path)
        
        #
        return render_template('preprocess.html')
    
    else:
        data_file_path = session.get('uploaded_data_file_path',None)
        uploaded_df = pd.read_csv(data_file_path)
        uploaded_df_html = uploaded_df.to_html()
        return render_template('preprocess.html')





@app.route('/showData')
def showCSV():
    data_file_path = session.get('uploaded_data_file_path',None)
    uploaded_df = pd.read_csv(data_file_path)
    uploaded_df_html = uploaded_df.to_html()

    return render_template('showCSV.html', data_var=uploaded_df_html)

if __name__ == "__main__":
    app.run(debug=True)