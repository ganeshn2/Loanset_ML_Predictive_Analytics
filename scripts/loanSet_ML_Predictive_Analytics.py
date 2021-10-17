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
from sklearn.metrics import classification_report,confusion_matrix,jaccard_score,\
    accuracy_score,f1_score,precision_score,recall_score


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

#
# def facet_grid(data,col,hue,palette,height, aspect,margin_titles,X,title):
#     plt.Figure(figsize=(12, 6), dpi=1000)
#     a = sns.FacetGrid(data, col, hue, palette, height, aspect,
#                       margin_titles)
#     a.map(plt.hist, 'X')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(title + ".jpg")
#     plt.show()
#
#
# def joint_plot(y,x,data,kind,title):
#     # a = sns.jointplot(y='age', x='Principal', data=loan_df, kind='scatter')
#     a = sns.jointplot(y, x, data, kind=kind)
#     plt.tight_layout()
#     plt.savefig(title + ".jpg")
#     plt.show()
#
#
# def pair_plot(df, hue, palette,title):
#     # grid = sns.pairplot(loan_df, hue='terms', palette='Set3')
#     grid = sns.pairplot(df, hue=hue, palette=palette)
#     grid = grid.map_upper(plt.scatter, color='darkred')
#     grid = grid.map_diag(plt.hist, bins=10, color='blue',
#                          edgecolor='k')
#     grid = grid.map_lower(sns.kdeplot, cmap='Reds')
#     plt.tight_layout()
#     plt.savefig(title + ".jpg")
#     plt.show()  # reference: https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166
#
#
# def kde_plot(x,y,cmap,a,b,c,d):
#     # kde = sns.kdeplot(loan_df5['terms'], loan_df5['age'],
#                 #cmap="plasma", shade=True, shade_lowest=False)
#     kde = sns.kdeplot(x,y,cmap = cmap, shade = True, shade_lowest = False)
#     plt.xlabel(x, fontsize=15)
#     plt.ylabel(y, fontsize=15)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.xlim(a,b)
#     plt.ylim(c,d)
#
#
# X = loan_df5.drop(['loan_status'], axis =1)
# y = loan_df5['loan_status']

if __name__ == "__main__":
    loan_df = loan_dataframe()
    # print(loan_dataframe().head())
    cat_to_cont = cat_to_cont(loan_df)
    print(cat_to_cont)
    # facet_grid(loan_df,Gender,loan_status,Set1,4,1.5,True,Principal)
    # facet_grid(loan_df,Gender,loan_status,Set1,4,1.5,True,age)
    # facet_grid(loan_df, Gender, loan_status, Set3, 4, 1.5, True,Dayofweek)
