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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from ydata_profiling import ProfileReport
from sklearn import preprocessing
from joblib import dump


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
             return redirect('/eda')

        elif request.form['submit_button'] == 'Manual Preprocess':
            #uploaded_preprocess_df = pd.read_csv(file_location)
            return redirect('/preprocess')
            #return render_template('preprocess.html', data_var = preprocess_df_html)
        
        elif request.form['submit_button'] == 'Auto Preprocess':
             return redirect('/autoprep')

        elif request.form['submit_button'] == 'Model':
             return render_template('model.html')
    
    else:
        return render_template('index.html')


@app.route('/eda', methods=['GET', 'POST'])
def eda():
    if request.method == 'GET':
        location_for_eda = session.get('uploaded_data_file_path',None)
        uploaded_eda_df = pd.read_csv(location_for_eda, skipinitialspace = True)
        if 'index' in uploaded_eda_df.columns:
            uploaded_eda_df.drop(labels=['index'], axis=1, inplace=True)
        
        else:
            pass

        #av = AutoViz_Class()
        #dft = av.AutoViz("", sep=",", depVar="", dfte=uploaded_eda_df, header=0, verbose=1, lowess=False, chart_format="server", max_rows_analyzed=150000, save_plot_dir=None)
        #small_eda_df = uploaded_eda_df.head(n=5)
        #eda_df_html = small_eda_df.to_html()

        p = ProfileReport(uploaded_eda_df, explorative=True, dark_mode=True)
        p.to_file("templates\eda.html")
        return render_template('eda.html')
    
    else:
        return render_template('eda.html')


#Manual preprocessing
@app.route('/preprocess', methods=['GET', 'POST'])
def Preprocess():
    if request.method == 'POST':
        
        data_file_path = session.get('uploaded_data_file_path',None)
        preprocess_df = pd.read_csv(data_file_path, skipinitialspace = True)

        if 'index' in preprocess_df.columns:
            preprocess_df.drop(labels=['index'], axis=1, inplace=True)
        
        else:
            pass

        #saving the target label of the dataset
        target_label = str(request.form['TargetLabel'])
        #target_label += ''.join((preprocess_df.iloc[[1],[len(col)-1]]).columns.tolist())

        #Columns to remove 
        columns_to_remove = str(request.form['ColumnsRemove']).split(',')
        if len(columns_to_remove)>1:
            for i in columns_to_remove:
                preprocess_df.drop(labels=[i], axis=1, inplace=True)
        else:
            pass

        col = preprocess_df.columns

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
            temp_target = preprocess_df.pop(target_label)

        
        elif request.form['MissingData'] == 'Imputation':
            preprocess_df = preprocess_df.drop(preprocess_df[preprocess_df[target_label] == 'NaN'].index)
            temp_target = preprocess_df.pop(target_label)
            col_list = preprocess_df.columns.values.tolist()
            imputer = KNNImputer(n_neighbors=5)
            imputed_data = imputer.fit_transform(preprocess_df)
            preprocess_df = pd.DataFrame(data=imputed_data, columns = col_list)

        elif request.form['MissingData'] == 'None':
            pass

        #Split data to make it easier to perform the functions below
        preprocess_df[target_label] = temp_target   #push target label back

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
        date_list = ['Day', 'Month', 'Year']
        if 'Day' in x_all_data.columns:     
            date_df = x_all_data[date_list].copy()
            x_all_data = x_all_data.drop(date_list, axis=1)
        else:
            date_list = ''
        

        new_col_list = preprocess_df.columns.to_list()
        target_label_list = list(target_label.split(" "))

        if request.form['ScaleData'] == 'Normalization':
            norm_var = MinMaxScaler().fit(x_all_data)
            x_all_data = norm_var.transform(x_all_data)

            exclude_list = [date_list, target_label]
            col_after_elimination = [i for i in new_col_list if i not in exclude_list]
            x_all_data = pd.DataFrame(data=x_all_data, columns = col_after_elimination)
            if 'Day' in x_all_data.columns:
                x_all_data = pd.concat([x_all_data,date_df], axis=1)
            else:
                pass
            

        elif request.form['ScaleData'] == 'Standardization':
            if not date_list:
                categorical_data = feature_for_label_enc + feature_for_one_enc + target_label_list
            else:
                categorical_data = feature_for_label_enc + feature_for_one_enc + date_list + target_label_list

            res = [i for i in new_col_list if i not in categorical_data]
            
            for j in res:
                temp_df = pd.DataFrame(x_all_data[j])
                stan_var = StandardScaler().fit(temp_df)
                x_all_data[j] = stan_var.transform(temp_df)
            if "Day" in x_all_data.columns:
                x_all_data = pd.concat([x_all_data,date_df], axis=1)
            else:
                pass

        elif request.form['ScaleData'] == 'None':
            pass

        #Correlation Based Feature selection
        preprocess_df.pop(target_label)
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

        #reverse label encoding
        if 'index' in preprocess_df:
            preprocess_df.drop(labels=['index'], axis=1, inplace=True)
        
        else:
            pass

        download_folder = os.path.join('static','downloads')
        download_file = 'Preprocessed_Data.csv'
        download_file_path = download_folder + "/" + download_file   #Creating path where the downloadable file will be stored
        preprocess_df.to_csv(download_file_path, index=False)
        session['download_data_file_path'] = download_file_path  #Store download file path in session
        return redirect('/download')
    
    elif request.method == 'GET':
        mylocation = session.get('uploaded_data_file_path',None)
        uploaded_preprocess_df = pd.read_csv(mylocation, skipinitialspace = True)

        if 'index' in uploaded_preprocess_df.columns:
            uploaded_preprocess_df.drop(labels=['index'], axis=1, inplace=True)
        else:
            pass

        small_preprocess_df = uploaded_preprocess_df.head(n=5)
        preprocess_df_html = small_preprocess_df.to_html()
        return render_template('preprocess.html', data_var = preprocess_df_html)


#Auto preprocessing
@app.route('/autoprep', methods=['GET', 'POST'])
def autoprep():
    if request.method == 'GET':
        mylocation = session.get('uploaded_data_file_path',None)
        uploaded_preprocess_df = pd.read_csv(mylocation, skipinitialspace = True)

        if 'index' in uploaded_preprocess_df.columns:
            uploaded_preprocess_df.drop(labels=['index'], axis=1, inplace=True)
        else:
            pass

        small_preprocess_df = uploaded_preprocess_df.head(n=5)
        preprocess_df_html = small_preprocess_df.to_html()
        return render_template('autoprep.html', data_var = preprocess_df_html)
    
    #Actual preprocessing begins here
    elif request.method == 'POST':
        data_file_path = session.get('uploaded_data_file_path',None)
        autoprep_df = pd.read_csv(data_file_path, skipinitialspace = True)

        if 'index' in autoprep_df.columns:
            autoprep_df.drop(labels=['index'], axis=1, inplace=True)
        
        else:
            pass

        #saving the target label of the dataset
        auto_target = str(request.form['TargetLabel'])
        #target_label += ''.join((preprocess_df.iloc[[1],[len(col)-1]]).columns.tolist())

        auto_col = autoprep_df.columns

        #Detecting and processing date time data
        feature_name = ''
        for i in range(1, len(auto_col)):
            if "/" in str(autoprep_df.iloc[[1],[i]]) and "Time" in str((autoprep_df.iloc[[1],[i]]).columns.tolist()) :
                var = autoprep_df.iloc[[1],[i]].columns.tolist()
                for item in var:
                    feature_name += item    
                autoprep_df[feature_name] = pd.to_datetime(autoprep_df[feature_name], format = '%d/%m/%Y %I:%M')
                autoprep_df['Day'] = autoprep_df[feature_name].dt.day
                autoprep_df['Month'] = autoprep_df[feature_name].dt.month
                autoprep_df['Year'] = autoprep_df[feature_name].dt.year
                autoprep_df.drop(autoprep_df.columns[[i]], axis=1, inplace=True)   #deleting the date time column

        #Identifying categorical features for encoding
        feature_for_one_enc = []
        for name, column in autoprep_df.items():
            unique_count = column.unique().shape[0]
            total_count = column.shape[0]
        if unique_count / total_count < 0.01:
            feature_for_one_enc.append(name)

        feature_for_one_enc = list(filter(lambda x: x != auto_target, feature_for_one_enc))
        date_list = ['Day', 'Month', 'Year']
        for i in date_list:
            feature_for_one_enc = list(filter(lambda x: x != i, feature_for_one_enc))
        
        #One hot encoding
        if not feature_for_one_enc:
            pass
        else:
            for y in feature_for_one_enc:
                autoprep_df = pd.get_dummies(autoprep_df, columns=[y], drop_first = True)
        
        #putting target label back at last
        autoprep_df.insert(len(autoprep_df.columns)-1, auto_target, autoprep_df.pop(auto_target))

        #Handling missing data through imputation      
        autoprep_df = autoprep_df.drop(autoprep_df[autoprep_df[auto_target] == 'NaN'].index)
        temp_target = autoprep_df.pop(auto_target)
        col_list = autoprep_df.columns.values.tolist()
        imputer = KNNImputer(n_neighbors=5)
        imputed_data = imputer.fit_transform(autoprep_df)
        autoprep_df = pd.DataFrame(data=imputed_data, columns = col_list)

        #Split data to make it easier to perform the functions below
        autoprep_df[auto_target] = temp_target   #push target label back

        X_train, X_test, y_train, y_test = train_test_split(
        autoprep_df.drop(labels=[auto_target], axis=1),
        autoprep_df[auto_target],
        test_size=0.3,
        random_state=None)

        x_all_data = pd.concat([X_train,X_test], axis=0)
        y_all_data = pd.concat([y_train,y_test], axis=0)

        #Handling imbalanced dataset
        sm = SMOTE(random_state=42)
        x_all_data, y_all_data = sm.fit_resample(x_all_data,y_all_data)
        

        #Feature scaling 
        
        if 'Day' in x_all_data.columns:     
            date_df = x_all_data[date_list].copy()
            x_all_data = x_all_data.drop(date_list, axis=1)
        else:
            date_list = ''
        

        new_col_list = autoprep_df.columns.to_list()

        if request.form['submit_button'] == 'KNN' or 'SVM':
            norm_var = MinMaxScaler().fit(x_all_data)
            x_all_data = norm_var.transform(x_all_data)

            list_tar = list(auto_target.split())
            if (len(date_list)>1):
                exclude_list = date_list + list_tar
            else:
                exclude_list = list_tar
                
            col_after_elimination = [i for i in new_col_list if i not in exclude_list]

            x_all_data = pd.DataFrame(data=x_all_data, columns = col_after_elimination)
            if 'Day' in x_all_data.columns:
                x_all_data = pd.concat([x_all_data,date_df], axis=1)
            else:
                pass

        else:
            pass

        #Correlation Based Feature selection
        autoprep_df.pop(auto_target)
        threshold = 0.90
        cor_features = set()   #set of all names of correlated columns
        cor_matrix = autoprep_df.corr()
        for i in range(len(cor_matrix.columns)):
            for j in range(i):
                if abs(cor_matrix.iloc[i, j]) > threshold: #Absolute coeff value is used
                    colName = cor_matrix.columns[i]  #saving the names of correlated columns
                    cor_features.add(colName)
        
        x_all_data.drop(labels=cor_features, axis=1, inplace=True)
        

        autoprep_df = pd.concat([x_all_data,y_all_data], axis=1)
        
        autoprep_df = autoprep_df.sample(frac=1, random_state=1).reset_index()

        #reverse label encoding
        if 'index' in autoprep_df:
            autoprep_df.drop(labels=['index'], axis=1, inplace=True)
        
        else:
            pass

        download_folder = os.path.join('static','downloads')
        download_file = 'Preprocessed_Data.csv'
        download_file_path = download_folder + "/" + download_file   #Creating path where the downloadable file will be stored
        autoprep_df.to_csv(download_file_path, index=False)
        session['download_data_file_path'] = download_file_path  #Store download file path in session
        return redirect('/download')
        


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


@app.route('/model', methods=['GET', 'POST'])
def model_implementation():
    if request.method == 'GET':
        return render_template('model.html')

    elif request.method == 'POST':
        data_file_for_model = session.get('uploaded_data_file_path', None)
        model_df = pd.read_csv(data_file_for_model, skipinitialspace=True)

        if 'index' in model_df:
            model_df.drop(labels=['index'], axis=1, inplace=True)

        model_target = model_df.columns[-1]     #Selecting the last label as target
        #train test split
        X_train, X_test, y_train, y_test = train_test_split(
        model_df.drop(labels=[model_target], axis=1),
        model_df[model_target],
        test_size=0.3,
        random_state=None)

        #Machine learning models
        
        if request.form['submit_button'] == 'Decision Tree':
            initialModel = DecisionTreeClassifier(
                            criterion='gini',
                            splitter='best'
                            )
            
            hyperparameters = { 'max_depth' : [i for i in range(1, 15)]}

            FinalModel = RandomizedSearchCV(
             initialModel,
             param_distributions = hyperparameters,
             cv = 10,
             n_jobs = -1
            )
        
            FinalModel.fit(X_train, y_train)
            
            best_params = FinalModel.best_params_
            
            Tuned_model = FinalModel.best_estimator_
        
        elif request.form['submit_button'] == 'KNN':
            hyperparameters = [{'leaf_size':[i for i in range(1, 20)], 
                    'n_neighbors':[j for j in range(1,30)],
                    'p':[1,2]}]
            
            initialModel = KNeighborsClassifier()
            
            FinalModel = RandomizedSearchCV(
            initialModel,
             param_distributions = hyperparameters,
             cv = 10,
             n_jobs = -1)
            
            FinalModel.fit(X_train, y_train)

            best_params = FinalModel.best_params_

            Tuned_model = FinalModel.best_estimator_   
        
        elif request.form['submit_button'] == 'SVM':
            hyperparameters = [{'kernel':['rbf'], 'C':[0.1,1,10,100], 
                    'gamma':[1,0.1,0.01,0.001]}]
            
            initialModel = SVC(random_state=1)

            FinalModel = RandomizedSearchCV(
             initialModel,
             param_distributions = hyperparameters,
             cv = 10,
             n_jobs = -1)
            
            FinalModel.fit(X_train, y_train)

            best_params = FinalModel.best_params_

            Tuned_model = FinalModel.best_estimator_

            
        predictions = Tuned_model.predict(X_test)
        report = classification_report(y_test, predictions, digits=4, output_dict=True)
        report_df = pd.DataFrame.from_dict(report)
        report_df = report_df.transpose()

        dump(Tuned_model,'static\models\model.joblib')
        model_file_path = 'static\models\model.joblib'
        session['model_file_path'] = model_file_path

        return render_template('result.html', result_data=report_df, params = best_params)
            
        
@app.route('/downloadModel', methods=['POST'])
def downloadModel():
    model_file_path = session.get('model_file_path',None)
    return send_file(model_file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)