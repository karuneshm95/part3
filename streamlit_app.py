
import streamlit as st

import pandas as pd

st.markdown("<h1 style='text-align: center; color: red;'>Fault Detector</h1>", unsafe_allow_html=True)
selected_file = st.sidebar.selectbox('Select Dataset', ('CWRU_12K_3hp.parquet', 'CWRU_12K_2hp.parquet', 'CWRU_12K_1hp.parquet','CWRU_12K_0hp.parquet','CWRU_48K_3hp.parquet', 'CWRU_48K_2hp.parquet', 'CWRU_48K_1hp.parquet','CWRU_48K_0hp.parquet'))
# Reset the warning filters after your Streamlit code

# Load the data
@st.cache
def load_data(selected_file):
    data = pd.read_parquet(selected_file)
    return data

data = load_data(selected_file)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
#import ipywidgets as widgets
import time
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, f_classif, mutual_info_classif, RFE, RFECV, SelectFromModel, VarianceThreshold, SelectFdr, SelectFpr, SelectFwe, SelectKBest, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
#from tpot import TPOTClassifier
#import warnings
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
#from skopt import BayesSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score
#import plotly.express as px
# Suppress the warning message



# Your Streamlit code
# ...

# Sidebar - Select Features and Labels

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], test_size=0.2, random_state=42)

# Split the data into train and test sets

selected_feature = st.sidebar.selectbox('Feature Selection', ('None', 'Chi-Squared','ANOVA_F-Valued',
             'Mutual_Information','RFE','RFECV','SFM',
            'Variance_Threshold','FDR','FPR','FWE'))
# Sidebar - Select Classifier
selected_classifier = st.sidebar.selectbox('Select Classifier', ('Logistic Regression', 'Random Forest', 'SVM','KNN','Decision Tree'))

selected_tuning = st.sidebar.selectbox('Hyperparameter Tuning', ('None', 'Grid Search', 'Random Search','Bayesian Optimization','Genetic Algorithm'))
def train_and_evaluate_model(model_name=selected_classifier,feature_selection=selected_feature, hyperparam_opt=selected_tuning):
    
    if model_name == 'Logistic Regression':
        model = LogisticRegression()
    elif model_name == 'Random Forest':
        model = RandomForestClassifier()
    elif model_name == 'SVM':
        model = SVC(probability=True)
    elif model_name =='KNN':
        model=KNeighborsClassifier()
    elif model_name =='Decision Tree':
        model=DecisionTreeClassifier()
    if feature_selection == 'ANOVA_F-Value':
        # Perform feature selection using SelectKBest
        selector = SelectKBest(score_func=f_classif, k=10)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
    elif feature_selection == 'Chi-Squared':
        # Perform feature selection using SelectKBest
        selector = SelectKBest(chi2, k=10)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
    elif feature_selection == 'Mutual_Information':
        # Perform feature selection using SelectKBest
        selector = SelectKBest(mutual_info_classif, k=2)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
    elif feature_selection == 'RFE':
        # Perform feature selection using SelectKBest
        selector = RFE(estimator=model,n_features_to_select=2)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
    elif feature_selection == 'RFE':
        # Perform feature selection using SelectKBest
        selector = RFE(estimator=model,min_features_to_select=2)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
    elif feature_selection == 'SFM':
        # Perform feature selection using SelectKBest
        selector = SelectFromModel(estimator=model,max_features=2)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
    elif feature_selection == 'Variance_Threshold':
        # Perform feature selection using SelectKBest
        selector = VarianceThreshold(threshold=0.1)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
    elif feature_selection == 'FDR':
        # Perform feature selection using SelectKBest
        selector = SelectFdr(f_classif,alpha=0.05)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
    elif feature_selection == 'FPR':
        # Perform feature selection using SelectKBest
        selector = SelectFpr(f_classif,alpha=0.05)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
    elif feature_selection == 'FWE':
        # Perform feature selection using SelectKBest
        selector = SelectFwe(f_classif,alpha=0.05)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
    else:
        X_train_selected = X_train
        X_test_selected = X_test


    if hyperparam_opt == 'Grid Search':
        if model==SVC():
            parameters = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001]}
            grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
            grid_search.fit(X_train_selected, y_train)

            # Get the best hyperparameters and evaluate on the test set
            best_grid_params = grid_search.best_params_
            model.set_params(**best_grid_params)
            model.fit(X_train_selected,y_train)
            
            start_time = time.time()
            model.predict(X_test_selected.sample(n=1))
            end_time = time.time()
            test_time_gsa = end_time - start_time
            return X_train_selected,X_test_selected,model,test_time_gsa,best_grid_params
        elif model==LogisticRegression():
            parameters={'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
            grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
            grid_search.fit(X_train_selected, y_train)

            # Get the best hyperparameters and evaluate on the test set
            best_grid_params = grid_search.best_params_
            model.set_params(**best_grid_params)
            model.fit(X_train_selected,y_train)
            
            start_time = time.time()
            model.predict(X_test_selected.sample(n=1))
            end_time = time.time()
            test_time_gsa = end_time - start_time
            return X_train_selected,X_test_selected,model,test_time_gsa,best_grid_params
        elif model==RandomForestClassifier():
            parameters={'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}
            grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
            grid_search.fit(X_train_selected, y_train)

            # Get the best hyperparameters and evaluate on the test set
            best_grid_params = grid_search.best_params_
            model.set_params(**best_grid_params)
            model.fit(X_train_selected,y_train)
            
            start_time = time.time()
            model.predict(X_test_selected.sample(n=1))
            end_time = time.time()
            test_time_gsa = end_time - start_time
            return X_train_selected,X_test_selected,model,test_time_gsa,best_grid_params
        elif model==KNeighborsClassifier():
            parameters={'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
            grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
            grid_search.fit(X_train_selected, y_train)

            # Get the best hyperparameters and evaluate on the test set
            best_grid_params = grid_search.best_params_
            model.set_params(**best_grid_params)
            model.fit(X_train_selected,y_train)
            
            start_time = time.time()
            model.predict(X_test_selected.sample(n=1))
            end_time = time.time()
            test_time_gsa = end_time - start_time
            return X_train_selected,X_test_selected,model,test_time_gsa,best_grid_params
        elif model==DecisionTreeClassifier():
            parameters={'max_depth': [None, 5, 10], 'criterion': ['gini', 'entropy']}
            grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
            grid_search.fit(X_train_selected, y_train)

            # Get the best hyperparameters and evaluate on the test set
            best_grid_params = grid_search.best_params_
            model.set_params(**best_grid_params)
            model.fit(X_train_selected,y_train)
            
            start_time = time.time()
            model.predict(X_test_selected.sample(n=1))
            end_time = time.time()
            test_time_gsa = end_time - start_time
            return X_train_selected,X_test_selected,model,test_time_gsa,best_grid_params
        else:
            model.fit(X_train_selected,y_train)
            best_grid_params=model.get_params()
            start_time = time.time()
            model.predict(pd.DataFrame(X_test_selected).sample(n=1))
            end_time = time.time()
            test_time_gsa = end_time - start_time
            return X_train_selected,X_test_selected,model,test_time_gsa,best_grid_params
            
        # Perform hyperparameter optimization using Grid Search
        
        # Your grid search code here
    elif hyperparam_opt == 'Random Search':
        if model==SVC():
            parameters = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001]}
            random_search = RandomizedSearchCV(model, param_distributions=parameters, cv=5)
            random_search.fit(X_train_selected, y_train)

            # Get the best hyperparameters and evaluate on the test set
            best_random_params = random_search.best_params_
            model.set_params(**best_random_params)
            model.fit(X_train_selected,y_train)
            
            start_time = time.time()
            model.predict(X_test_selected.sample(n=1))
            end_time = time.time()
            test_time_rs = end_time - start_time
            return X_train_selected,X_test_selected,model,test_time_rs,best_random_params
        elif model==LogisticRegression():
            parameters={'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
            random_search = RandomizedSearchCV(model, param_distributions=parameters, cv=5)
            random_search.fit(X_train_selected, y_train)

            # Get the best hyperparameters and evaluate on the test set
            best_random_params = random_search.best_params_
            model.set_params(**best_random_params)
            model.fit(X_train_selected,y_train)
            
            start_time = time.time()
            model.predict(X_test_selected.sample(n=1))
            end_time = time.time()
            test_time_rs = end_time - start_time
            return X_train_selected,X_test_selected,model,test_time_rs,best_random_params
        elif model==RandomForestClassifier():
            parameters={'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}
            random_search = RandomizedSearchCV(model, param_distributions=parameters, cv=5)
            random_search.fit(X_train_selected, y_train)

            # Get the best hyperparameters and evaluate on the test set
            best_random_params = random_search.best_params_
            model.set_params(**best_random_params)
            model.fit(X_train_selected,y_train)
            
            start_time = time.time()
            model.predict(X_test_selected.sample(n=1))
            end_time = time.time()
            test_time_rs = end_time - start_time
            return X_train_selected,X_test_selected,model,test_time_rs,best_random_params
        elif model==KNeighborsClassifier():
            parameters={'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
            random_search = RandomizedSearchCV(model, param_distributions=parameters, cv=5)
            random_search.fit(X_train_selected, y_train)

            # Get the best hyperparameters and evaluate on the test set
            best_random_params = random_search.best_params_
            model.set_params(**best_random_params)
            model.fit(X_train_selected,y_train)
            
            start_time = time.time()
            model.predict(X_test_selected.sample(n=1))
            end_time = time.time()
            test_time_rs = end_time - start_time
            return X_train_selected,X_test_selected,model,test_time_rs,best_random_params
        elif model==DecisionTreeClassifier():
            parameters={'max_depth': [None, 5, 10], 'criterion': ['gini', 'entropy']}
            random_search = RandomizedSearchCV(model, param_distributions=parameters, cv=5)
            random_search.fit(X_train_selected, y_train)

            # Get the best hyperparameters and evaluate on the test set
            best_random_params = random_search.best_params_
            model.set_params(**best_random_params)
            model.fit(X_train_selected,y_train)
            
            start_time = time.time()
            model.predict(X_test_selected.sample(n=1))
            end_time = time.time()
            test_time_rs = end_time - start_time
            return X_train_selected,X_test_selected,model,test_time_rs,best_random_params
        else:
            model.fit(X_train_selected,y_train)
            best_random_params=model.get_params()
            start_time = time.time()
            model.predict(pd.DataFrame(X_test_selected).sample(n=1))
            end_time = time.time()
            test_time_rs = end_time - start_time
            return X_train_selected,X_test_selected,model,test_time_rs,best_random_params
            
    elif hyperparam_opt =='Bayesian Optimization':
        if model==SVC():
            parameters = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001]}
            bayes_search = BayesSearchCV(model, param_space, cv=5, n_iter=50)
            bayes_search.fit(X_train_selected, y_train)

            # Get the best hyperparameters and evaluate on the test set
            best_bayes_params = bayes_search.best_params_
            model.set_params(**best_bayes_params)
            model.fit(X_train_selected, y_train)
            
            start_time = time.time()
            model.predict(X_test_selected.sample(n=1))
            end_time = time.time()
            test_time_bs = end_time - start_time
            return X_train_selected,X_test_selected,model,test_time_bs,best_bayes_params
        elif model==LogisticRegression():
            parameters={'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
            bayes_search = BayesSearchCV(model, param_space, cv=5, n_iter=50)
            bayes_search.fit(X_train_selected, y_train)

            # Get the best hyperparameters and evaluate on the test set
            best_bayes_params = bayes_search.best_params_
            model.set_params(**best_bayes_params)
            model.fit(X_train_selected, y_train)
            
            start_time = time.time()
            model.predict(X_test_selected.sample(n=1))
            end_time = time.time()
            test_time_bs = end_time - start_time
            return X_train_selected,X_test_selected,model,test_time_bs,best_bayes_params
        elif model==RandomForestClassifier():
            parameters={'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}
            bayes_search = BayesSearchCV(model, param_space, cv=5, n_iter=50)
            bayes_search.fit(X_train_selected, y_train)

            # Get the best hyperparameters and evaluate on the test set
            best_bayes_params = bayes_search.best_params_
            model.set_params(**best_bayes_params)
            model.fit(X_train_selected, y_train)
            
            start_time = time.time()
            model.predict(X_test_selected.sample(n=1))
            end_time = time.time()
            test_time_bs = end_time - start_time
            return X_train_selected,X_test_selected,model,test_time_bs,best_bayes_params
        elif model==KNeighborsClassifier():
            parameters={'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
            bayes_search = BayesSearchCV(model, param_space, cv=5, n_iter=50)
            bayes_search.fit(X_train_selected, y_train)

            # Get the best hyperparameters and evaluate on the test set
            best_bayes_params = bayes_search.best_params_
            model.set_params(**best_bayes_params)
            model.fit(X_train_selected, y_train)
            
            start_time = time.time()
            model.predict(X_test_selected.sample(n=1))
            end_time = time.time()
            test_time_bs = end_time - start_time
            return X_train_selected,X_test_selected,model,test_time_bs,best_bayes_params
        elif model==DecisionTreeClassifier():
            parameters={'max_depth': [None, 5, 10], 'criterion': ['gini', 'entropy']}
            bayes_search = BayesSearchCV(model, param_space, cv=5, n_iter=50)
            bayes_search.fit(X_train_selected, y_train)

            # Get the best hyperparameters and evaluate on the test set
            best_bayes_params = bayes_search.best_params_
            model.set_params(**best_bayes_params)
            model.fit(X_train_selected, y_train)
            
            start_time = time.time()
            model.predict(X_test_selected.sample(n=1))
            end_time = time.time()
            test_time_bs = end_time - start_time
            return X_train_selected,X_test_selected,model,test_time_bs,best_bayes_params
        else:
            model.fit(X_train_selected, y_train)
            best_bayes_params=model.get_params()
            start_time = time.time()
            model.predict(X_test_selected.sample(n=1))
            end_time = time.time()
            test_time_bs = end_time - start_time
            return X_train_selected,X_test_selected,model,test_time_bs,best_bayes_params
        # Perform hyperparameter optimization using Random Search
    else:
        model.fit(X_train_selected, y_train)
        
        start_time = time.time()
        
        model.predict(pd.DataFrame(X_test_selected).sample(n=1))
        
        end_time = time.time()
        test_time = end_time - start_time
        best_params=model.get_params()
        return X_train_selected,X_test_selected,model,test_time,best_params
        # Your random search code here

    
    # Fit the model on the training data
    

    # Print the results
    
    
X_train_selected,X_test_selected,best_estimator,test_time,best_params=train_and_evaluate_model(selected_classifier,selected_feature,selected_tuning)

# Display Accuracy and Classification Report
accuracy_train=accuracy_score(y_train,best_estimator.predict(X_train_selected))
accuracy_test=accuracy_score(y_test,best_estimator.predict(X_test_selected))
st.write(f"Train Accuracy:{accuracy_train}")
st.write(f"Test Accuracy:{accuracy_test}")
st.write(f"Parameters are:{best_params}")
st.write(f"Execution Time is:{test_time}")
cr_test=classification_report(y_test,best_estimator.predict(X_test_selected))
st.subheader('Classification Report')
st.code(cr_test)
cm = confusion_matrix(y_test, best_estimator.predict(X_test_selected))

# Compute ROC AUC
roc_auc = roc_auc_score(y_test, best_estimator.predict_proba(X_test_selected), multi_class='ovr')


# Display confusion matrix
st.subheader('Confusion Matrix')
st.write(pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1','Predicted 2','Predicted 3'], index=['Actual 0', 'Actual 1','Actual 2','Actual 3']))

# Display ROC AUC
st.subheader('ROC AUC')
st.write(f"ROC AUC: {roc_auc}")
from sklearn.metrics import roc_curve
# Plot ROC curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve

# Convert integer labels to one-hot encoded format
lb = LabelBinarizer()
y_test_encoded = lb.fit_transform(y_test)

# Get the predicted probabilities for each class
y_scores = best_estimator.predict_proba(X_test_selected)
num_classes=4
# Calculate the ROC curve for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_encoded[:, i], y_scores[:, i])

# Plot the ROC curve for each class (example for one class)
st.line_chart(pd.DataFrame({'False Positive Rate': fpr[0], 'True Positive Rate': tpr[0]}))
import scipy.io
file = st.sidebar.selectbox('Select Test Dataset', ('100.mat','112.mat','125.mat','138.mat'))
mat_data = scipy.io.loadmat(file)
if file=='100.mat':
        data=mat_data['X100_DE_time']#healthy
        df = pd.DataFrame(data,columns=['DE'])
        df=df.iloc[:2000,:]
        df['Category']=0
if file=='112.mat':
        data=mat_data['X112_DE_time']#inner
        df = pd.DataFrame(data,columns=['DE'])
        df['Category']=1
if file=='125.mat':
        data=mat_data['X125_DE_time']#ball
        df = pd.DataFrame(data,columns=['DE'])
        df['Category']=2
if file=='138.mat':
        data=mat_data['X138_DE_time']#outer
        df = pd.DataFrame(data,columns=['DE'])
        df['Category']=3
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import scipy.signal as signal
import pywt
import scipy.stats as stats
from scipy.stats import skew, kurtosis
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import welch
import pywt
#df=pd.read_parquet('vibration.parquet') 

import pandas as pd


# getting excel files to be merged from the Desktop 


# read all the files with extension .xlsx i.e. excel 

# Create an empty list to store the dictionaries
data = []

    
        
   
splits = np.array_split(df.iloc[:20000,:], 100)
for split in splits:
        
    x=split['DE'].to_numpy()
    y=split['Category'].to_numpy()
    nperseg=min(len(x),256)
    freq_spectrum=np.fft.fft(x)
    freq_amplitude=np.abs(freq_spectrum)
    frequencies_Ax,power_spectrum_Ax=welch(x)
    freq_power_DnI_1=np.square(freq_amplitude)
    freq_skewness_DnI_1=skew(freq_amplitude)
    freq_kurtosis_DnI_1=kurtosis(freq_amplitude)
    frequencies,time,spectrogram=signal.spectrogram(x,fs=48000,nperseg=nperseg)
    tf_skewness=skew(spectrogram)
    tf_kurtosis=kurtosis(spectrogram)
        
    row ={
                            'median_DnI_1':np.median(x),
                            'Kurtosis_value_DnI_1':kurtosis(x),
                             'Skewness_DnI_1':skew(x),
                             
                             
                             'mean_freq_skewness_DnI_1':np.mean(freq_skewness_DnI_1),
                             
                              
                              'tf_min':np.min(spectrogram),
                              
                               
                            
                             
                             
                              'std_tf_skewness':np.std(tf_skewness),
                               'sum_tf_skewness':np.sum(tf_skewness),
              
                            
                              'std_tf_kurtosis':np.std(tf_kurtosis),
                             
                             'max_pf_DnI_1':frequencies_Ax[np.argmax(power_spectrum_Ax)],
                              'total_power_DnI_1':np.sum(power_spectrum_Ax),
                             
                              
                            
                             
                             
                            'vibration_Category':int(np.mean(split['Category']))}
    data.append(row)

    
# Create a DataFrame from the list of dictionaries
df_vibrationa_bAx = pd.DataFrame(data)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the existing file
#data = pd.read_csv('existing_data.csv')  # Assuming the data is in CSV format

# Calculate the correlation matrix


# Print the highly correlated feature pairs



# Plot correlation in a scatter plot

pred=best_estimator.predict(df_vibrationa_bAx.iloc[:,:-1])
st.write(f"Test Prediction:{pred}")
accuracy_test=accuracy_score(df_vibrationa_bAx.iloc[:,-1],best_estimator.predict(df_vibrationa_bAx.iloc[:,:-1]))
st.write(f"Test Dataset Accuracy:{accuracy_test}")
