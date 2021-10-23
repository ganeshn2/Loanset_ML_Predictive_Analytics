import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,jaccard_score,accuracy_score,f1_score,precision_score,recall_score


def loan_dataframe():
    loan_df1 = pd.read_csv('./input/Loanset.csv', index_col=[0])
    loan_df1 = loan_df1.drop(columns=['Unnamed: 0.1', 'Unnamed: 0.1.1'], axis=1)
    loan_df1['effective_date'] = pd.to_datetime(loan_df1['effective_date'])
    loan_df1['due_date'] = pd.to_datetime(loan_df1['due_date'])
    loan_df1['Dayofweek'] = loan_df1['due_date'].dt.dayofweek
    loan_df1 = loan_df1.drop(['effective_date', 'due_date'], axis=1)
    loan_df1['weekend'] = loan_df1['Dayofweek'].apply(lambda x: 1 if (x > 4) else 0)
    loan_df1["education"] = loan_df1["education"].replace({"Bechalor": "Bachelor"})
    return loan_df1


def cat_to_cont(df):
    loan_df2 = pd.get_dummies(loan_df1['education'], drop_first=False)
    loan_df3 = pd.get_dummies(loan_df1['Gender'],drop_first=False)
    loan_df4 = pd.concat([loan_df2, loan_df3], axis=1)
    return loan_df4


def clean_df(df1, df2):
    loan_df = pd.concat([loan_df1, loan_df4], axis=1)
    loan_df = loan_df.drop(['education', 'Gender'], axis=1)
    return loan_df

def error_rate():
    error_rate = []
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        prediction_i = knn.predict(X_train)
        error_rate.append(np.mean(prediction_i != y_train))
    min_val = min(error_rate)
    min_index = error_rate.index(min_val)
    return min_index+1


def ml_model(ml_alg_type):
    if ml_alg_type == "LogisticRegression":
        regression = LogisticRegression()
    elif ml_alg_type == "KNeighborsClassifier":
        regression = KNeighborsClassifier(error_rate())
    elif ml_alg_type == "DecisionTreeClassifier":
        regression = DecisionTreeClassifier()
    elif ml_alg_type == "RandomForestClassifier":
        regression = RandomForestClassifier(criterion='gini',n_estimators=10,verbose=3)
    elif ml_alg_type == "SVC":
        regression = SVC(kernel='rbf',degree=3)
    elif ml_alg_type == "GaussianNB":
        regression = GaussianNB(priors=None, var_smoothing=1e-09)
    regression.fit(X_train, y_train)
    predictions_title = regression.predict(X_test)
    ps_title = precision_score(y_test, predictions_title, average='weighted').round(2)
    rs_title = recall_score(y_test, predictions_title, average='weighted').round(2)
    js_title = jaccard_score(y_test, predictions_title, average='weighted').round(2)
    f1s_title = f1_score(y_test, predictions_title, average='weighted').round(2)
    cr = classification_report(y_test, predictions_title)
    cm = confusion_matrix(y_test, predictions_title)
    # print("Classification Report Using Log Model for the Dataset is:", '\n\n', cr)
    # print("Confusion Matrix Using Log Model for the Dataset is:",'\n\n',cm)
    return {
        "precision_score":ps_title,
        "recall_score": rs_title,
        "jaccuard_score":js_title,
        "f1_score":f1s_title
        }


if __name__ == "__main__":
    loan_df1 = loan_dataframe()
    loan_df4 = cat_to_cont(loan_df1)
    loan_df = clean_df(loan_df1,loan_df4)
    X = loan_df.drop(['loan_status'], axis=1)
    y = loan_df['loan_status']
    X = StandardScaler().fit(X).transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=101)
    alg_list = ['LogisticRegression','KNeighborsClassifier',
                'DecisionTreeClassifier','RandomForestClassifier','SVC','GaussianNB']
    result_dict = dict()
    for alg in alg_list:
        result = ml_model(alg)
        result_dict[alg] = result
    scores_df = pd.DataFrame(result_dict).T
    print(scores_df)
    #print(error_rate())