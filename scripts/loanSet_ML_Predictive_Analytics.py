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
    loan_df = pd.read_csv('./input/Loanset.csv', index_col=[0])
    loan_df = loan_df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0.1.1'], axis=1)
    loan_df['effective_date'] = pd.to_datetime(loan_df['effective_date'])
    loan_df['due_date'] = pd.to_datetime(loan_df['due_date'])
    loan_df['Dayofweek'] = loan_df['due_date'].dt.dayofweek
    loan_df = loan_df.drop(['effective_date', 'due_date'], axis=1)
    loan_df['weekend'] = loan_df['Dayofweek'].apply(lambda x: 1 if (x > 4) else 0)
    return loan_df


def cat_to_cont(df):
    df1 = pd.get_dummies(loan_df['education'], drop_first=False)
    df2 = pd.get_dummies(loan_df['Gender'],drop_first=True)
    df3 = pd.concat([df1, df2], axis=1)
    return df3




if __name__ == "__main__":
    loan_dataframe()
    print(loan_dataframe(loan_df))
    cat_to_cont()