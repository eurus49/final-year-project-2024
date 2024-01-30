from flask import Flask, render_template,redirect, session, url_for, request
from werkzeug.utils import secure_filename
import pandas as pd
from fileinput import filename
import os
from sklearn.impute import KNNImputer          #importing KNNImputer for imputation of missing values
from sklearn import preprocessing
#from script import detect_data                #Importing function to detect data types


UPLOAD_FOLDER = os.path.join('static','uploads')

ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'Secret key to utilize session'

@app.route('/', methods=['GET', 'POST'])
def uploapreprocess_dfile():
    if request.method == 'POST':
        #Upload file
        uploadepreprocess_dfile = request.files['file']  

        #Extracting uploaded file name
        data_filename = secure_filename(uploadepreprocess_dfile.filename)
        
        file_location = os.path.join((app.config['UPLOAD_FOLDER']),data_filename)
        
        uploadepreprocess_dfile.save(file_location)          

        session['uploaded_data_file_path'] = file_location

        if request.form['submit_button'] == 'EDA':
            return render_template('eda.html')
        
        elif request.form['submit_button'] == 'Preprocess':
            #uploaded_preprocess_df = pd.read_csv(file_location)
            return redirect('/preprocess')
            #return render_template('preprocess.html', data_var = preprocess_df_html)
    
    else:
        return render_template('index.html')

@app.route('/preprocess', methods=['GET', 'POST'])
def Preprocess():
    if request.method == 'POST':
        
        data_file_path = session.get('uploaded_data_file_path',None)
        preprocess_df = pd.read_csv(data_file_path, skipinitialspace = True)
        col = preprocess_df.columns

        #saving the target label of the dataset
        target_label = ''
        target_label += ''.join((preprocess_df.iloc[[1],[len(col)-1]]).columns.tolist())
        
        #Detecting and processing date time data
        feature_name = ''
        for i in range(1, len(col)):
            if "/" in str(preprocess_df.iloc[[1],[i]]) and "Time" in str((preprocess_df.iloc[[1],[i]]).columns.tolist()) :
                var = preprocess_df.iloc[[1],[i]].columns.tolist()
                for item in var:
                    feature_name += item    
                preprocess_df[feature_name] = pd.to_datetime(preprocess_df[feature_name], format = '%d/%m/%Y %I:%M')
                preprocess_df['Day'] = preprocess_df[feature_name].dt.day
                preprocess_df['Month'] = preprocess_df[feature_name].dt.month
                preprocess_df['Year'] = preprocess_df[feature_name].dt.year
                preprocess_df.drop(preprocess_df.columns[[i]], axis=1, inplace=True)   #deleting the date time column

        #Label encoding
        feature_for_label_enc = str(request.form['LabelEncoding']).split()
        for x in feature_for_label_enc:
            label_encoder = preprocessing.LabelEncoder()
            preprocess_df[x] = label_encoder.fit_transform(preprocess_df[x])

        #One hot encoding
        feature_for_one_enc = str(request.form['OneHotEncoding']).split()
        for y in feature_for_one_enc:
            preprocess_df = pd.get_dummies(preprocess_df, columns=[y], drop_first = True)
        
        #putting target label back at last
        preprocess_df.insert(len(preprocess_df.columns)-1, target_label, preprocess_df.pop(target_label))  

        
        #Handling missing data
        if request.form['MissingData'] == 'Deletion':
            preprocess_df.dropna(inplace = True)
            preprocess_df.drop_duplicates(inplace = True)
        
        elif request.form['MissingData'] == 'Imputation':
            preprocess_df = preprocess_df.drop(preprocess_df[preprocess_df[target_label] == 'NaN'].index)
            col_list = preprocess_df.columns.values.tolist()
            imputer = KNNImputer(n_neighbors=5)
            imputed_data = imputer.fit_transform(preprocess_df)
            preprocess_df = pd.DataFrame(data=imputed_data, columns = col_list )


        
        uploaded_preprocess_df_html = preprocess_df.to_html()
        return render_template('PrepSuccess.html', var_newdata=uploaded_preprocess_df_html)
    
    elif request.method == 'GET':
        mylocation = session.get('uploaded_data_file_path',None)
        uploaded_preprocess_df = pd.read_csv(mylocation)
        small_preprocess_df = uploaded_preprocess_df.head(n=5)
        preprocess_df_html = small_preprocess_df.to_html()
        return render_template('preprocess.html', data_var = preprocess_df_html)





@app.route('/showData')
def showCSV():
    data_file_path = session.get('uploaded_data_file_path',None)
    uploaded_preprocess_df = pd.read_csv(data_file_path)
    uploaded_preprocess_df_html = uploaded_preprocess_df.to_html()

    return render_template('showCSV.html', data_var=uploaded_preprocess_df_html)

if __name__ == "__main__":
    app.run(debug=True)