#import the dependencies
import pandas as  pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    data = read_data()
    print("================================================================")
    preview_data(data)
    print("===========*****Wait-->Plotting the Data*****==============================================")
    plot_columns(data)
    print("================================================================")
    features, labels = labelAndFeatures(data)
    print("================================================================")
    features_encoded = encode_data(features)
    print("================================================================")
    features_scaled_df = scale_data(features_encoded)
    print("================================================================")
    x_train, x_test, y_train, y_test = split_data(features_scaled_df, labels, split_size=0.3)
    print("================***Training the Data***============================================")
    models = training_the_data(x_train, y_train)
    print("================================================================")
    # print(type(models))
    predict_results(models, x_test, y_test)
    dump_model(models)

#loading the data
def read_data():
    data = pd.read_csv('diabetes.csv')
    # return data
    return data    

def preview_data(data):
    print(data.head()) #preview of data
    #shape of the data
    print("Shape of the data:", data.shape)

#Columns of the data
    print("Columns of the data: ", data.columns)

#total-info of the data
    print("Info:", data.info())

#Description of the data
    print("Description of the data:", data.describe())

#Plotting the data
def plot_columns(data):
    data.hist(bins=500, figsize=(20, 10))
    plt.show()

def scale_data(data):
#Scaling the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    # extract numerical attributes and scale it to have zero mean and unit variance  
    cols = data.select_dtypes(include=['float64','int64']).columns
    sc_train = scaler.fit_transform(data.select_dtypes(include=['float64','int64']))


    # turn the result back to a dataframe
    sc_traindf = pd.DataFrame(sc_train, columns = cols)

    return sc_traindf

def labelAndFeatures(data):
    #Splitting the class Label from the data
    y = data['Outcome']
    x = data.drop(['Outcome'], axis=1)
    return x, y

def encode_data(features):
#encoding the data
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    for i in features.columns:
        features[i] = encoder.fit_transform(features[i])
    return features

def split_data(x, y, split_size):
    #splitting the train and test data
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_size) #70 percent for training

    return x_train, x_test, y_train, y_test


def training_the_data(x_train, y_train):
    #Loading all the Classifiers
    rf = RandomForestClassifier(n_estimators = 1000, random_state=7)
    svm = SVC(kernel='linear')
    abc = AdaBoostClassifier(n_estimators=100)
    dt = DecisionTreeClassifier(max_depth=3, min_samples_split=2)
    knn = KNeighborsClassifier(n_neighbors=3)
    gnb = GaussianNB()

    rf_clf = rf.fit(x_train, y_train) #training the Random-Forest
    svm_clf = svm.fit(x_train, y_train) #training the SVM
    abc_clf = abc.fit(x_train, y_train) #training the AdaBoost
    dt_clf = dt.fit(x_train, y_train) #training the Decision Tree
    knn_clf = knn.fit(x_train, y_train) #training the KNN
    gnb_clf = gnb.fit(x_train, y_train) #training the Naive Bayes

    return rf_clf, svm_clf, abc_clf, dt_clf, knn_clf, gnb_clf


def predict_results(models, x_test, y_test):
    classifiers = ["Random Forest", "Support Vector Machine", "AdaBoost", "Decision Tree", "K-NN", "Naive Bayes"]
    j=0
    for i in models:
        predict = i.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, predict)
        print("Accuracy for : ", classifiers[j], accuracy*100)
        j = j + 1
        confusion_matrix = metrics.confusion_matrix(y_test, i.predict(x_test))
        print("Confusion Matrix:")
        print(confusion_matrix)

def dump_model(models):
    joblib.dump(models[0], "Models/RF.pkl")
    joblib.dump(models[1], "SVM.pkl")
    joblib.dump(models[2], "Models/Ada.pkl")
    joblib.dump(models[3], "Models/DT.pkl")
    joblib.dump(models[4], "Models/KNN.pkl")
    joblib.dump(models[5], "Models/NB.pkl")
if __name__ == '__main__':
    main()
    