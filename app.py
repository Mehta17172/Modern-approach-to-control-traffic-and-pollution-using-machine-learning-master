import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)
#model = pickle.load(open('model_sdl.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    dataset = pd.read_csv("data.csv")
    X = dataset.iloc[: , [0,1,2]].values
    y = dataset.iloc[: , -1].values
    
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.17,random_state = 0)
    
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.fit_transform(X_test)
    
    #fitting KNN regression
    
    from sklearn.neighbors import KNeighborsClassifier
    classifier =  KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
    classifier.fit(X_train,y_train)
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    y_pred = classifier.predict(sc_X.transform(final_features))
    
    if y_pred == 1:
        vehicle = "BICYCLE"
    elif y_pred == 2:
        vehicle = "BIKE"
    elif y_pred == 3:
        vehicle = "CAR"
    
    return render_template('main.html', prediction_text='Suitable Vehicle for you is {}'.format(vehicle))

if __name__ == "__main__":
    app.run(debug=True)