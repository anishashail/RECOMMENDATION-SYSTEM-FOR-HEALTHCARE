#import the standard gui(graphical user interface) library for python
import tkinter as tki

#import all the classes and functions from tkinter module
from tkinter import *

#import themed tkinter module which provides additional themed widgets. Normal Tkinter.* widgets are not themed!
from tkinter import ttk

#import the 'messagebox' class from tkinter used for displaying pop up message boxes
from tkinter import messagebox

#import the 'font' class from tkinter used for customizing fonts in tkinter widgets
from tkinter.font import Font

#import the 'Image' and 'ImageTk' classes from the Python Imaging Library (PIL). This is used for handling images.
from PIL import Image, ImageTk

#import thr 'numpy' and 'pandas' library for calculation, data manipulation and analysis.
import numpy as np
import pandas as pd

#import the RandomForestClassifier class from the sklearn (scikit-learn) library. 
from sklearn.ensemble import RandomForestClassifier

#import specific functions for evaluating the performance of the machine learning model, such as accuracy, classification report, and confusion matrix.
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

#import the 'train_test_split' function from scikit-learn
from sklearn.model_selection import train_test_split

#import 'themed_tk' which allows you to provide additional themes for tkinter widgets
from ttkthemes import themed_tk as tk

root = tk.ThemedTk()
root.set_theme("black")
image = Image.open("C:/Users/anish/OneDrive/Desktop/MINI_PROJECT_5TH_SEM/Files/wh.png")
bgImage = ImageTk.PhotoImage(image)
Label(root , image = bgImage).place(relwidth= 1, relheight= 1)
root.title("Specialist Recommendation Tool")
root.configure()

class Window(tki.Toplevel):  #The parent class makes an independent window
    def __init__(self, parent):

        df = pd.read_csv("C:/Users/anish/OneDrive/Desktop/MINI_PROJECT_5TH_SEM/Dataset/Symptom.csv")
        df.drop(['Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9', 'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13',
        'Symptom_14', 'Symptom_15', 'Symptom_16', 'Symptom_17'], axis=1, inplace=True)   #axis = 1 means work along the columns,
                                                                                        #changes are happening inplace, i.e in same file
        cols = df.columns
        data = df[cols].values.flatten()   #make 1-d numpy array
        s = pd.Series(data)    #create labeled pandas series
        s = s.str.strip()   #remove whitespaces from strings in the series
        s = s.values.reshape(df.shape)    #reshape as original dataframe
        df = pd.DataFrame(s, columns=df.columns)   #replace the original dataframe with the modified dataframe
        df = df.fillna(0)   #fill missing value with 0
        vals = df.values     #numpy array

        df2 = pd.read_csv("C:/Users/anish/OneDrive/Desktop/MINI_PROJECT_5TH_SEM/Dataset/Disease Specialist.csv")
        specialist = df2['Specialist'].tolist()
        edd = df2['Disease'].tolist()

        df1 = pd.read_csv("C:/Users/anish/OneDrive/Desktop/MINI_PROJECT_5TH_SEM/Dataset/Symptom Severity.csv")
        symptoms = df1['Symptom'].unique()
        for i in range(len(symptoms)):
            vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]    #replaces the matching symptoms in vals with their
                                                                                                     #weights in severity
            
        d = pd.DataFrame(vals, columns=cols)
        d = d.replace('dischromic _patches', 6)
        d = d.replace('spotting_ urination', 6)
        df = d.replace('foul_smell_of urine', 5)

        data = df.iloc[:, 1:].values   #numpy array from  all rows and column index 1
        labels = df['Disease'].values    #create a series of first column or disease names

        # Data splitting for training and testing
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)   #20 percent for testing 

        # Random Forest model training
        rf = RandomForestClassifier(n_estimators=100)    #n_estimators gives the number of decision trees
        rf = rf.fit(train_data, train_labels)    #training happens here
        Label(root , image = bgImage).place(relwidth= 1, relheight= 1)
        root.title("Specialist Recommendation Tool")
        root.configure()

        # Testing the model on the test set
        predictions = rf.predict(test_data)

        # Model evaluation
        accuracy = accuracy_score(test_labels, predictions)
        print(f"Accuracy: {accuracy}")     #accuracy value inserted in the curly braces

        Symptom1 = StringVar()
        Symptom1.set(None)
        Symptom2 = StringVar()
        Symptom2.set(None)
        Symptom3 = StringVar()
        Symptom3.set(None)
        Symptom4 = StringVar()
        Symptom4.set(None)
        Symptom5 = StringVar()
        Symptom5.set(None)

        fgColor = "#000000"
        bgColor = "Alice Blue"
        w2 = Label(root, justify=CENTER, text=" Specialist Recommendation Tool", fg=fgColor , bg= bgColor)

        w2.config(font=("Harrington", 25))
        w2.grid(row=1, column=0, columnspan=2, padx=100)

        NameLb1 = Label(root, text="")
        NameLb1.config(font=("Bookman Old Style", 20))
        NameLb1.grid(row=5, column=1, pady=10, sticky=W)

        S1Lb = Label(root, text="Symptom 1", fg=bgColor, bg = fgColor)
        S1Lb.config(font=("Bookman Old Style", 15))
        S1Lb.grid(row=7, column=1, pady=10, sticky=W)

        S2Lb = Label(root, text="Symptom 2", fg=bgColor, bg = fgColor)
        S2Lb.config(font=("Bookman Old Style", 15))
        S2Lb.grid(row=8, column=1, pady=10, sticky=W)

        S3Lb = Label(root, text="Symptom 3", fg=bgColor, bg = fgColor)
        S3Lb.config(font=("Bookman Old Style", 15))
        S3Lb.grid(row=9, column=1, pady=10, sticky=W)

        S4Lb = Label(root, text="Symptom 4", fg=bgColor, bg = fgColor)
        S4Lb.config(font=("Bookman Old Style", 15))
        S4Lb.grid(row=10, column=1, pady=10, sticky=W)

        S5Lb = Label(root, text="Symptom 5", fg=bgColor, bg = fgColor)
        S5Lb.config(font=("Bookman Old Style", 15))
        S5Lb.grid(row=11, column=1, pady=10, sticky=W)

        OPTIONS = df1['Symptom']

        S1En = ttk.OptionMenu(root, Symptom1, *OPTIONS)
        S1En.grid(row=7, column=1)

        S2En = ttk.OptionMenu(root, Symptom2, *OPTIONS)
        S2En.grid(row=8, column=1)

        S3En = ttk.OptionMenu(root, Symptom3, *OPTIONS)
        S3En.grid(row=9, column=1)

        S4En = ttk.OptionMenu(root, Symptom4, *OPTIONS)
        S4En.grid(row=10, column=1)

        S5En = ttk.OptionMenu(root, Symptom5, *OPTIONS)
        S5En.grid(row=11, column=1)

        NameLb = Label(root, text="")
        NameLb.config(font=("Bookman Old Style", 20))
        NameLb.grid(row=13, column=1, pady=10, sticky=W)

        NameLb = Label(root)
        NameLb.config(font=("Bookman Old Style", 15))
        NameLb.grid(row=17, column=1, pady=10, sticky=W)

        t4 = Text(root, height=2, width=20)
        t4.config(font=("Bookman Old Style", 20))
        t4.grid(row=20, column=1, padx=10)


        def RF():
            psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
            a = np.array(df1["Symptom"])
            b = np.array(df1["weight"])
            for j in range(len(psymptoms)):
                for k in range(len(a)):
                    if psymptoms[j] == a[k]:
                        psymptoms[j] = b[k]
            psy = [psymptoms]
            prob = rf.predict(psy)
            print(prob[0])
            spec = prob[0]
            for l in range(len(edd)):
                if spec == edd[l]:
                    t4.delete("1.0", END)
                    t4.insert(END, specialist[l])


        def message():
            if (Symptom1.get() == None and Symptom2.get() == None and Symptom3.get() == None and Symptom4.get() == None
                    and Symptom5.get() == None):
                messagebox.showinfo("OPPS!!", "ENTER  SYMPTOMS PLEASE")
            else:
                RF()

        
        lr = ttk.Button(root, text="Predict", command=message)
        lr.grid(row=15, column=1, pady=10)

        b = ttk.Button(root, text="Close", command=root.destroy)
        b.grid(row=21, column=1, padx=5, pady=10)



class MyTk(tki.Tk): #MyTk is inheriting from tki.Tk
    def __init__(self):
        super().__init__()

        self.geometry('300x200')
        self.title('Main Window')
        
        # place a button on the root window
        ttk.Button(self, text='Open The Recommendation System', command=self.open_window).pack(expand=True)

    def open_window(self):
        window = Window(self)  


if __name__ == "__main__":


    #initialise a main window        
    app = MyTk()

    #set the window background color as black
    app.configure(background = "#000000")

    app.mainloop()
