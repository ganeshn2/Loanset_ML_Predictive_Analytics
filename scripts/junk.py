# facet_grid(loan_df1,loan_df1['Gender'],loan_df1['loan_status'],palette,loan_df1['Principal'],"Gender")
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

colors_1 = ["#008fd5","#fc4f30","#e5ae37","#6d904f","#D02090","#308014"]
colors_2 = ["#008fd5","#fc4f30","#e5ae37","#6d904f","#D02090"]
labels_1 = ['North America',"Europe","Asia","South America","Africa","Oceania"]
labels_2 = ['Device',"Drug","Other","Procedure","Biological"]
explode_1 = [0.1, 0, 0, 0, 0.2, 0]
explode_2 = [0.1, 0, 0, 0, 0.1]


def pie_chart(slices,labels, colors, explode,title):
    plt.style.use("fivethirtyeight")
    plt.pie(slices, labels=labels, explode=explode,
            startangle=-15, autopct="%1.0f%%",
            colors=colors,
            wedgeprops={"edgecolor": "black"})
    plt.title(title, fontsize=18)
    plt.tight_layout()
    plt.savefig(title+".jpg")
    plt.show()
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


    # facet_grid(loan_df,Gender,loan_status,Set1,4,1.5,True,age)
    # facet_grid(loan_df, Gender, loan_status, Set3, 4, 1.5, True,Dayofweek)

# def facet_grid(data,col,hue,palette,X,title):
#     a = sns.FacetGrid(data=data, col=col, hue=hue, palette=palette, aspect=aspect,
#                       margin_titles=True)
#     a.map(plt.hist, X)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(title + ".jpg")
#     plt.show()

# aspect = 1.5
# palette = ['Set1']