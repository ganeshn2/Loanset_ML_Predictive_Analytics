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

def loan_dataframe():
    df = pd.read_csv('./input/Loanset.csv', index_col=[0])
    df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0.1.1'], axis=1)
    df['effective_date'] = pd.to_datetime(df['effective_date'])
    df['due_date'] = pd.to_datetime(df['due_date'])
    df['Dayofweek'] = df['due_date'].dt.dayofweek
    df = df.drop(['effective_date', 'due_date'], axis=1)
    df['weekend'] = df['Dayofweek'].apply(lambda x: 1 if (x > 4) else 0)
    return df


def cat_to_cont(df):
    df1 = pd.get_dummies(df['education','Gender'],drop_first = False)
    return df1



if __name__ == "__main__":
    loan_dataframe()
    print(loan_dataframe().head())
    #cat_to_cont(df)