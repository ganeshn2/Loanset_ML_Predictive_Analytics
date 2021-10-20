import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
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


# def ml_models(title):
#     logmodel = LogisticRegression()
#     logmodel.fit(X_train, y_train)
#     predictions_log = logmodel.predict(X_test)
#     ps_title= precision_score(y_test,predictions_log,average='weighted').round(2)
#     rs_title=recall_score(y_test,predictions_log,average='weighted').round(2)
#     js_title=jaccard_score(y_test,predictions_log,average='weighted').round(2)
#     f1s_title=f1_score(y_test,predictions_log,average='weighted').round(2)
#     cr_title = classification_report(y_test, predictions_log)
#     cm_title = confusion_matrix(y_test,predictions_log)
#     print("Classification Report Using Log Model for the Dataset is:",'\n\n', cr_log)
#     print("Confusion Matrix Using Log Model for the Dataset is:",'\n\n',cm_log)
#     return ps_title,rs_log,js_log,f1s_log


if __name__ == "__main__":
    loan_df1 = loan_dataframe()
    loan_df4 = cat_to_cont(loan_df1)
    loan_df = clean_df(loan_df1,loan_df4)
    X = loan_df.drop(['loan_status'], axis=1)
    y = loan_df['loan_status']
    X = StandardScaler().fit(X).transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=101)
    print("success")