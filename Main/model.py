
from replace import *


# Decision tree is classified as a very effective and versatile classification technique. It is used in pattern recognition and classification for image. It is used for classification in very complex problems dew to its high adaptability. It is also capable of engaging problems of higher dimensionality. It mainly consists of three parts root, nodes and leaf.
# Roots consists of attribute which has most effect on the outcome, leaf tests for value of certain attribute and leaf gives out the output of tree.


def DecisionTree():

    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()
    clf3 = clf3.fit(X, y)

    from sklearn.metrics import accuracy_score
    y_pred = clf3.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred, normalize=False))

    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(),
                 Symptom4.get(), Symptom5.get()]

    for k in range(0, len(l1)):
        for z in psymptoms:
            if(z == l1[k]):
                l2[k] = 1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted = predict[0]

    h = 'no'
    for a in range(0, len(disease)):
        if(predicted == a):
            h = 'yes'
            break

    if (h == 'yes'):
        t1.delete("1.0", END)
        t1.insert(END, disease[a])
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")
    import sqlite3 
    conn = sqlite3.connect('database.db') 
    c = conn.cursor() 
    c.execute("CREATE TABLE IF NOT EXISTS DecisionTree(Name StringVar,Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 TEXT,Symtom5 TEXT,Disease StringVar)")
    c.execute("INSERT INTO DecisionTree(Name,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?,?)",(NameEn.get(),Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get(),t1.get("1.0",END)))
    conn.commit()  
    c.close() 
    conn.close()


# Random Forest Algorithm is a supervised learning algorithm used for both classification and regression. This algorithm works on 4 basic steps

# 1. It chooses random data samples from dataset.

# 2. It constructs decision trees for every sample dataset chosen.

# 3. At this step every predicted result will be compiled and voted on.

# 4. At last most voted prediction will be selected and be presented as result of classification.

# In this project we have used random forest classifier with 100 random samples and the result given is -95 % accuracy.

def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X, np.ravel(y))

    # calculating accuracy
    from sklearn.metrics import accuracy_score
    y_pred = clf4.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred, normalize=False))

    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(),
                 Symptom4.get(), Symptom5.get()]

    for k in range(0, len(l1)):
        for z in psymptoms:
            if(z == l1[k]):
                l2[k] = 1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted = predict[0]

    h = 'no'
    for a in range(0, len(disease)):
        if(predicted == a):
            h = 'yes'
            break

    if (h == 'yes'):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")
        
    import sqlite3
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS RandomForest(Name StringVar,Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 TEXT,Symtom5 TEXT,Disease StringVar)")
    c.execute("INSERT INTO RandomForest(Name,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?,?)",
              (NameEn.get(), Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get(), t2.get("1.0",END)))
    conn.commit()
    c.close()
    conn.close()


# Naïve Bayes algorithm is a family of algorithms based on naïve bayes theorem. They share a common principle that is every pair of prediction is independent of each other. It also makes an assumption that features make an independent and equal contribution to the prediction.

# In our project we have used naïve bayes algorithm to gain a - 95 % accurate prediction.
def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb = gnb.fit(X, np.ravel(y))

    from sklearn.metrics import accuracy_score
    y_pred = gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred, normalize=False))

    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(),
                 Symptom4.get(), Symptom5.get()]
    for k in range(0, len(l1)):
        for z in psymptoms:
            if(z == l1[k]):
                l2[k] = 1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted = predict[0]

    h = 'no'
    for a in range(0, len(disease)):
        if(predicted == a):
            h = 'yes'
            break

    if (h == 'yes'):
        t3.delete("1.0", END)
        t3.insert(END, disease[a])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")
        
    import sqlite3
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS NaiveBayes(Name StringVar,Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 TEXT,Symtom5 TEXT,Disease StringVar)")
    c.execute("INSERT INTO NaiveBayes(Name,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?,?)",
              (NameEn.get(), Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get(), t3.get("1.0", END)))
    conn.commit()
    c.close()
    conn.close()

    
    
        
#  K Nearest Neighbour is a supervised learning algorithm. It is a basic yet essential algorithm. It finds extensive use in pattern finding and data mining.

# It works by finding a pattern in data which links data to results and it improves upon the patter recognition with every iteration.

# We have used K Nearest Neighbour to classify our dataset and achieved -92% accuracy.   
    
def Knn():
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    knn = knn.fit(X, np.ravel(y))

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    y_pred = knn.predict(X_test)
    print("kNearest Neighbour")
    print("Accuracy")
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred, normalize=False))
    
    
        

    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(),
                     Symptom4.get(), Symptom5.get()]

    for k in range(0, len(l1)):
        for z in psymptoms:
            if(z == l1[k]):
                l2[k] = 1
    
    inputtest = [l2]
    predict = knn.predict(inputtest)
    predicted = predict[0]

    h = 'no'
    for a in range(0, len(disease)):
        if(predicted == a):
            h = 'yes'
            break

    if (h == 'yes'):
        t4.delete("1.0", END)
        t4.insert(END, disease[a])
    else:
        t4.delete("1.0", END)
        t4.insert(END, "Not Found")
        
    import sqlite3
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS KNearestNeighbour(Name StringVar,Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 TEXT,Symtom5 TEXT,Disease StringVar)")
    c.execute("INSERT INTO KNearestNeighbour(Name,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?,?)",
              (NameEn.get(), Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get(), t4.get("1.0",END)))
    conn.commit()
    c.close()
    conn.close()
         
 
def predictDisease(symptoms):
    
    
    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": nb_prediction,
        "final_prediction": final_prediction
}
    return predictions


# GUI stuff..............................................................................

root = Tk()
root.configure(background='BLACK')

Symptom1 = StringVar()
Symptom1.set("Select Here")

Symptom2 = StringVar()
Symptom2.set("Select Here")

Symptom3 = StringVar()
Symptom3.set("Select Here")

Symptom4 = StringVar()
Symptom4.set("Select Here")

Symptom5 = StringVar()
Symptom5.set("Select Here")

Name = StringVar()

w2 = Label(root, justify=LEFT,
           text="Dr.ONLINE", fg="red", bg="White")
w2.config(font=("Times", 30, "bold italic"))
w2.grid(row=1, column=0, columnspan=2, padx=100)


NameLb = Label(root, text="Name of the Patient", fg="Red", bg="Sky Blue")
NameLb.config(font=("Times", 15, "bold italic"))
NameLb.grid(row=6, column=0, pady=15, sticky=W)

S1Lb = Label(root, text="Symptom 1", fg="Blue", bg="Pink")
S1Lb.config(font=("Times", 15, "bold italic"))
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2", fg="White", bg="Purple")
S2Lb.config(font=("Times", 15, "bold italic"))
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="Green", bg="white")
S3Lb.config(font=("Times", 15, "bold italic"))
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="blue", bg="Yellow")
S4Lb.config(font=("Times", 15, "bold italic"))
S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="purple", bg="light green")
S5Lb.config(font=("Times", 15, "bold italic"))
S5Lb.grid(row=11, column=0, pady=10, sticky=W)


lrLb = Label(root, text="DecisionTree", fg="white", bg="red")
lrLb.config(font=("Times", 15, "bold italic"))
lrLb.grid(row=15, column=0, pady=10, sticky=W)

destreeLb = Label(root, text="RandomForest", fg="Red", bg="Orange")
destreeLb.config(font=("Times", 15, "bold italic"))
destreeLb.grid(row=17, column=0, pady=10, sticky=W)

ranfLb = Label(root, text="NaiveBayes", fg="White", bg="green")
ranfLb.config(font=("Times", 15, "bold italic"))
ranfLb.grid(row=19, column=0, pady=10, sticky=W)

ranfLb = Label(root, text="KNN", fg="Blue", bg="yellow")
ranfLb.config(font=("Times", 15, "bold italic"))
ranfLb.grid(row=21, column=0, pady=10, sticky=W)

OPTIONS = sorted(l1)

NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)

S1 = OptionMenu(root, Symptom1, *OPTIONS)
S1.grid(row=7, column=1)

S2 = OptionMenu(root, Symptom2, *OPTIONS)
S2.grid(row=8, column=1)

S3 = OptionMenu(root, Symptom3, *OPTIONS)
S3.grid(row=9, column=1)

S4 = OptionMenu(root, Symptom4, *OPTIONS)
S4.grid(row=10, column=1)

S5 = OptionMenu(root, Symptom5, *OPTIONS)
S5.grid(row=11, column=1)


dst = Button(root, text="Prediction 1",
             command=DecisionTree, bg="Red", fg="yellow")
dst.config(font=("Times", 15, "bold italic"))
dst.grid(row=8, column=3, padx=10)

rnf = Button(root, text="Prediction 2",
             command=randomforest, bg="White", fg="green")
rnf.config(font=("Times", 15, "bold italic"))
rnf.grid(row=9, column=3, padx=10)

lr = Button(root, text="Prediction 3",
            command=NaiveBayes, bg="Blue", fg="white")
lr.config(font=("Times", 15, "bold italic"))
lr.grid(row=10, column=3, padx=10)

lr = Button(root, text="Prediction 4",
            command= Knn, bg="yellow", fg="blue")
lr.config(font=("Times", 15, "bold italic"))
lr.grid(row=11, column=3, padx=10)


t1 = Text(root, height=1, width=40, bg="Light green", fg="red")
t1.config(font=("Times", 15, "bold italic"))
t1.grid(row=15, column=1, padx=10)

t2 = Text(root, height=1, width=40, bg="White", fg="Blue")
t2.config(font=("Times", 15, "bold italic"))
t2.grid(row=17, column=1, padx=10)

t3 = Text(root, height=1, width=40, bg="red", fg="white")
t3.config(font=("Times", 15, "bold italic"))
t3.grid(row=19, column=1, padx=10)

t4 = Text(root, height=1, width=40, bg="purple", fg="white")
t4.config(font=("Times", 15, "bold italic"))
t4.grid(row=21, column=1, padx=10)




root.mainloop()
