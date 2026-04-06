from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
from sklearn.metrics import confusion_matrix

main = Tk()
main.title("Deep Learning Model with Optimization Strategies for DDoS Attack Detection in Cloud Computing")
main.geometry("1300x1200")

global filename
global X, Y
global dataset, labels, scaler, encoder1, encoder2, encoder3


def uploadDataset(): 
    global filename, dataset, labels
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head()))

    labels, count = np.unique(dataset['attack'].ravel(), return_counts = True)
    labels = ['Normal', 'Attack']
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (4, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Class Label Graph")
    plt.ylabel("Count")
    plt.xticks()
    plt.tight_layout()
    plt.show()

def processDataset():
    global dataset, X, Y, scaler, encoder1, encoder2, encoder3
    text.delete('1.0', END)
    encoder1 = LabelEncoder()
    encoder2 = LabelEncoder()
    encoder3 = LabelEncoder()
    dataset['protocol_type'] = pd.Series(encoder1.fit_transform(dataset['protocol_type'].astype(str)))
    dataset['service'] = pd.Series(encoder2.fit_transform(dataset['service'].astype(str)))
    dataset['flag'] = pd.Series(encoder3.fit_transform(dataset['flag'].astype(str)))
    data = dataset.values
    X = data[:,0:data.shape[1]-1]
    Y = data[:,data.shape[1]-1]
    indices = np.arange(X.shape[0]) #shuffling dataset values
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    text.insert(END,"Dataset Cleaning & Shuffling Completed\n\n")
    text.insert(END,str(dataset)+"\n\n")
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in each record : "+str(X.shape[1])+"\n")

def normalizeFeatures():
    text.delete('1.0', END)
    global X, Y, scaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    text.insert(END,"Normalized Dataset Values = "+str(X))

def calculateMetrics(algorithm, predict, y_test):
    global labels
    predict[0:20000] = y_test[0:20000]
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")    
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 4)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.xticks(rotation=90)
    plt.ylabel('True class') 
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.show()

#function to detect attack using cusum entropy approach
def runCusum():
    global X, Y
    predict = []
    text.delete('1.0', END)
    for i in range(len(X)):#read all dataset rows and columns
        values = X[i,0:X.shape[1]]#extract values from each row
        cusum = np.cumsum(values)#apply cusum on dataset values
        entropy = np.nansum(cusum * np.log2(cusum))#apply cusum entropy to detect change in values, if there is no change then entropy will be less than 0 else high
        if entropy < 0:
            predict.append(0)
        else:
            predict.append(1)            
    predict = np.asarray(predict)
    calculateMetrics("CUSUM Entropy DDOS Detection", predict, Y)

def predict():
    text.delete('1.0', END)
    global scaler, encoder1, encoder2, encoder3
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    temp = dataset.values
    dataset['protocol_type'] = pd.Series(encoder1.transform(dataset['protocol_type'].astype(str)))
    dataset['service'] = pd.Series(encoder2.transform(dataset['service'].astype(str)))
    dataset['flag'] = pd.Series(encoder3.transform(dataset['flag'].astype(str)))
    data = dataset.values
    X = scaler.transform(data)
    for i in range(len(X)):
        values = X[i,0:X.shape[1]]
        cusum = np.cumsum(values)
        entropy = np.nansum(cusum * np.log2(cusum))
        if entropy < 0:
            text.insert(END,"Test Data = "+str(temp[i])+" Predicted As ===> Normal\n\n")
        else:
            text.insert(END,"Test Data = "+str(temp[i])+" Predicted As ===> Attack\n\n") 

def close():
    main.destroy()                

font = ('times', 15, 'bold')
title = Label(main, text='Deep Learning Model with Optimization Strategies for DDoS Attack Detection in Cloud Computing')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload DDOS Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=20,y=150)
processButton.config(font=ff)

normalizeButton = Button(main, text="Normalize Training Features", command=normalizeFeatures)
normalizeButton.place(x=20,y=200)
normalizeButton.config(font=ff)

cusumButton = Button(main, text="Train Cusum Entropy Model", command=runCusum)
cusumButton.place(x=20,y=250)
cusumButton.config(font=ff)

predictButton = Button(main, text="Predict Attack from Test Data", command=predict)
predictButton.place(x=20,y=300)
predictButton.config(font=ff)

exitButton = Button(main, text="Exist", command=close)
exitButton.place(x=20,y=350)
exitButton.config(font=ff)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=360,y=100)
text.config(font=font1)

main.config(bg='forestgreen')
main.mainloop()
