import os
import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv


def step1(excel_cols):
    # 1. Get all abstracts in a dataframe
    if not os.path.isfile("step1.tsv"):
        print("Step 1")
        df = pd.read_csv(filepath_or_buffer="test_20240208.tsv", sep="\t")
        df = df.loc[:, excel_cols]
        df.to_csv("step1.tsv", sep="\t", index=False)
        print(df.head(5))


def step2():
    # 2. Embed each abstract and store the resulttuples (type, text, embedding)
    if not os.path.isfile("embeddings.tsv"):
        print("Step 2")
        load_dotenv()
        df = pd.read_csv(filepath_or_buffer="step1.tsv", sep="\t")
        embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", client=None)
        embeddings_list = []
        cntabstract = 0
        abstract_list = df['abstract'].tolist()
        abstract_list = [str(i) for i in abstract_list]
        for abstract in abstract_list:
            if cntabstract % 500 == 0:
                print(cntabstract)
            if abstract is not None and abstract != "" and abstract != "nan":
                embedding = embeddings_model.embed_query(abstract)
                embeddings_list.append(embedding)
            else:
                print(f"found empty abstract at row nr {cntabstract + 2}")
                embeddings_list.append(None)
            cntabstract += 1
        df['embedding'] = embeddings_list
        df[["title", "embedding"]].to_csv("embeddings.tsv", sep="\t", index=False)
        print(df.head(5))


def step3():
    # 3. Merge the file from step 1 and the embeddings file
    if not os.path.isfile("step3.tsv"):
        print("Step 3")
        df_emb = pd.read_csv(filepath_or_buffer="embeddings.tsv", sep="\t")
        df = pd.read_csv(filepath_or_buffer="step1.tsv", sep="\t")
        df_merged = pd.merge(df_emb, df, on='title', how='inner')
        print(f"total nr of papers: {df_merged.shape[0]}")
        # exclude the observations where no abstract is available
        df_merged = df_merged.loc[df_merged["embedding"].notnull(), :]
        print(f"total nr of papers with abstracts: {df_merged.shape[0]}")
        df_merged.to_csv("step3.tsv", sep="\t", index=False)
        print(df_merged.head(5))


def step4():
    # 4. Convert the 1536-dimensional lists into separate columns
    if not os.path.isfile("step4.tsv"):
        print("Step 4")
        df = pd.read_csv(filepath_or_buffer="step3.tsv", sep="\t")
        df_exp = pd.DataFrame(df['embedding'].str.split().values.tolist())
        # Rename embedding values columns
        df_exp.columns = ["dim_" + str(i) for i in range(1536)]
        # remove leading square bracket from first column and trailing square bracket from last column
        df_exp["dim_0"] = df_exp["dim_0"].str[1:]
        df_exp["dim_1535"] = df_exp["dim_1535"].str[:-1]
        # remove trailing commas from all values
        df_exp = df_exp.map(lambda x: x[:-1] if isinstance(x, str) else x)
        # convert all values to floats
        df_exp = df_exp.astype(np.float64)
        df_exp.to_csv("step4.tsv", sep="\t", index=False)
        print(df_exp.head(5))


def step5(excel_cols):
    # 5. Use T-SNE algorithm to reduce dimensions to 2 (or 3)
    if not os.path.isfile("step5.tsv"):
        print("Step 5")
        df = pd.read_csv(filepath_or_buffer="step3.tsv", sep="\t")
        df_exp = pd.read_csv(filepath_or_buffer="step4.tsv", sep="\t")
        tsne2 = TSNE(random_state=0, n_components=2)
        tsne2_results = tsne2.fit_transform(df_exp)
        tsne2_results = pd.DataFrame(tsne2_results, columns=['tsne1', 'tsne2'])
        # Combine the new columns with the original DataFrame
        dftsne2 = pd.concat([df[excel_cols], tsne2_results], axis=1)
        # for plotting: sort dataframe on "core" values low to high, NA's first
        dftsne2 = dftsne2.sort_values(by="core", na_position="first")
        dftsne2.to_csv("step5.tsv", sep="\t", index=False)
        print(dftsne2.head(5))


def step6():
    # 6. Visualize plots
    # Source: https://stackoverflow.com/questions/7908636/how-to-add-hovering-annotations-to-a-plot
    print("Step 6")
    dftsne2 = pd.read_csv(filepath_or_buffer="step5.tsv", sep="\t")
    # manipulate values of "included" column
    dftsne2.loc[dftsne2["missed"] == 1, "included"] = -1
    dftsne2.loc[dftsne2["included"].isnull(), "included"] = 0

    c = dftsne2["included"]
    norm = plt.Normalize(-1, 1)
    colors = ['red', 'grey', 'green']
    cmap = ListedColormap(colors, 'indexed')

    # create "relevancy" column:
    # value = 0 if irrelevant = 1 and JB = null
    # value = 1 if irrelevant = 1 and JB = 1
    # value = 2 if irrelevant = null and non-core = 1
    # value = 3 if irrelevant = null and core = 1
    # value = 4 in all other cases
    dftsne2["relevancy"] = np.where(dftsne2["core"] == 0, 0,
                                    np.where(dftsne2["core"] == 1, 1, 2))

    dftsne2["relevancy_string"] = np.where(dftsne2["core"] == 0, "non-core",
                                           np.where(dftsne2["core"] == 1, "core", "other"))

    dftsne2["annotation"] = dftsne2["relevancy_string"] + ": " + dftsne2["title"]
    names = dftsne2["annotation"]

    colors_dict = {-1: "red", 0: "grey", 1: "green"}
    included_dict = {-1: "miss", 0: "", 1: "hit"}
    colors_list = [-1, 0, 1]

    edgecolor_dict = {0: "orange", 1: "green", 2: "grey"}
    relevancy_dict = {0: "non-core", 1: "core", 2: ""}
    edgecolor_list = [0, 1, 2]
    edgecolors = tuple(map(edgecolor_dict.get, dftsne2["relevancy"]))

    plt.style.use('dark_background')

    fig, ax = plt.subplots()
    sc = plt.scatter(x=dftsne2["tsne1"], y=dftsne2["tsne2"], s=40, c=c, cmap=cmap, norm=norm, edgecolors=edgecolors)

    # Create custom legend for colors
    color_legend_handles = [plt.Line2D([0], [0], marker='o', color='black', markerfacecolor=colors_dict[item],
                                       markersize=10, label=included_dict[item]) for item in colors_list]
    first_legend = ax.legend(handles=color_legend_handles, title="", loc=1)

    # # Add the legend manually to the Axes.
    ax.add_artist(first_legend)

    # Create custom legend for edgecolors
    edgecolor_legend_handles = [plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='none',
                                           markeredgecolor=edgecolor_dict[item], markersize=10,
                                           label=relevancy_dict[item]) for item in edgecolor_list]
    plt.title("Final selection of full text papers, text-embedding-ada-002 embeddings")
    ax.legend(handles=edgecolor_legend_handles, title="", loc=4)

    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}".format("\n".join([names[n] for n in ind["ind"]]))
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        annot.get_bbox_patch().set_alpha(1.0)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()


# Steps:
def main():
    excel_cols = ["title", "abstract", "included", "missed", "core"]
    step1(excel_cols)
    step2()
    step3()
    step4()
    step5(excel_cols)
    step6()


if __name__ == "__main__":
    main()
