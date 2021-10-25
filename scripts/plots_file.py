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

def facet_grid(data,col,hue,palette):
    plt.Figure(figsize=(12, 6), dpi=1000)
    a = sns.FacetGrid(data=loan_df1, col='Gender', hue='loan_status', palette='Set1', height=6, aspect=1.5,
                      margin_titles=True)
    a.map(plt.hist, 'Principal')
    a.legend()
    a.show()



if __name__ == "__main__":
