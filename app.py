#from script import detect_data 
from flask import Flask, render_template,redirect, send_file, session, url_for, request
from werkzeug.utils import secure_filename
import pandas as pd
from fileinput import filename
import os
from sklearn.impute import KNNImputer          #importing KNNImputer for imputation of missing values
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
              

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
        preprocess_df.drop(labels=['index'], axis=1, inplace=True)
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
        if not feature_for_label_enc:
            pass
        else:
            for x in feature_for_label_enc:
                label_encoder = preprocessing.LabelEncoder()
                preprocess_df[x] = label_encoder.fit_transform(preprocess_df[x])

        #One hot encoding
        feature_for_one_enc = str(request.form['OneHotEncoding']).split()
        if not feature_for_one_enc:
            pass
        else:
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
            preprocess_df = pd.DataFrame(data=imputed_data, columns = col_list)

        elif request.form['MissingData'] == 'None':
            pass

        #Split data to make it easier to perform the functions below
        X_train, X_test, y_train, y_test = train_test_split(
        preprocess_df.drop(labels=[target_label], axis=1),
        preprocess_df[target_label],
        test_size=0.3,
        random_state=None)

        x_all_data = pd.concat([X_train,X_test], axis=0)
        y_all_data = pd.concat([y_train,y_test], axis=0)

        #Handling imbalanced dataset
        if request.form['ImbalanceData'] == 'UnderSample':
            rs = RandomUnderSampler(random_state=42)
            x_all_data, y_all_data = rs.fit_resample(x_all_data,y_all_data)
        
        elif request.form['ImbalanceData'] == 'OverSample':
            sm = SMOTE(random_state=42)
            x_all_data, y_all_data = sm.fit_resample(x_all_data,y_all_data)
        
        elif request.form['ImbalanceData'] == 'None':
            pass

        #Feature scaling 
        date_df = x_all_data[['Day', 'Month', 'Year']].copy()
        x_all_data = x_all_data.drop(['Day', 'Month', 'Year'], axis=1)
        
        new_col_list = preprocess_df.columns.to_list()

        if request.form['ScaleData'] == 'Normalization':
            norm_var = MinMaxScaler().fit(x_all_data)
            x_all_data = norm_var.transform(x_all_data)

            new_target = preprocess_df.columns[-1]
            exclude_list = ['Day', 'Month', 'Year', new_target]
            col_after_elimination = [i for i in new_col_list if i not in exclude_list]
            x_all_data = pd.DataFrame(data=x_all_data, columns = col_after_elimination)
            x_all_data = pd.concat([x_all_data,date_df], axis=1)
            

        elif request.form['ScaleData'] == 'Standardization':
            date_list = ['Day', 'Month', 'Year']
            categorical_data = feature_for_label_enc + feature_for_one_enc + date_list
            res = [i for i in new_col_list if i not in categorical_data]
            
            for j in res:
                temp_df = pd.DataFrame(x_all_data[j])
                stan_var = StandardScaler().fit(temp_df)
                x_all_data[j] = stan_var.transform(temp_df)
            x_all_data = pd.concat([x_all_data,date_df], axis=1)

        elif request.form['ScaleData'] == 'None':
            pass

        #Correlation Based Feature selection
        threshold = 0.90
        cor_features = set()   #set of all names of correlated columns
        cor_matrix = preprocess_df.corr()
        for i in range(len(cor_matrix.columns)):
            for j in range(i):
                if abs(cor_matrix.iloc[i, j]) > threshold: #Absolute coeff value is used
                    colName = cor_matrix.columns[i]  #saving the names of correlated columns
                    cor_features.add(colName)
        
        x_all_data.drop(labels=cor_features, axis=1, inplace=True)
        

        preprocess_df = pd.concat([x_all_data,y_all_data], axis=1)
        preprocess_df = preprocess_df.sample(frac=1, random_state=1).reset_index()

        download_folder = os.path.join('static','downloads')
        download_file = 'Preprocessed_Data.csv'
        download_file_path = download_folder + "/" + download_file
        preprocess_df.to_csv(download_file_path, index=False)
        session['download_data_file_path'] = download_file_path
        return redirect('/download')
    
    elif request.method == 'GET':
        mylocation = session.get('uploaded_data_file_path',None)
        uploaded_preprocess_df = pd.read_csv(mylocation, skipinitialspace = True)
        small_preprocess_df = uploaded_preprocess_df.head(n=5)
        preprocess_df_html = small_preprocess_df.to_html()
        return render_template('preprocess.html', data_var = preprocess_df_html)



@app.route('/download', methods=['GET', 'POST'])
def download():
    if request.method == 'GET':
        download_file_path = session.get('download_data_file_path',None)
        download_preprocess_df = pd.read_csv(download_file_path)
        temp_down = download_preprocess_df.head(10)
        download_preprocess_df_html = temp_down.to_html()

        return render_template('PrepSuccess.html', var_newdata=download_preprocess_df_html)

    elif request.method == 'POST':
        return redirect('/download-csv')

@app.route('/download-csv')
def download_csv():
    download_file_path = session.get('download_data_file_path',None)
    return send_file(download_file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)