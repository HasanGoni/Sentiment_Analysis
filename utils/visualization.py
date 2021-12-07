import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.patches import Rectangle
from wordcloud import WordCloud, STOPWORDS
from utils.preprocess_utils import review_to_words


def plot_embeddings(embedding,
                    labels,
                    tokenizer):
    """Plot embedding colored by label

    Args:
        embedding ([Vector]): [embedding of a sentence]
        labels ([list of str]): [name of the label]
    """

    fig = plt.figure(figsize=(16, 10))
    color_map = {'positive': "green", 'negative': 'red', 'neutral': 'black'}
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[color_map[x] for x in labels],
        s=40,
        alpha=0.4
    )
    handles = [Rectangle((0, 0), 1, 1, color=c, ec='k')
               for c in ['green', 'red', 'black']]
    labels = ['positive', 'negative', 'neutral']
    plt.legend(handles, labels)
    plt.title(f'Embedding Representation of {tokenizer} tokenizer ')
    plt.gca().set_aspect('equal', 'box')
    plt.gca().set_xlabel('x')
    plt.gca().set_ylabel('y')


def plot_embedding_cluster(vectorized,
                           umap_emb):
    """Genrate kMeans clustering
    and plotting them based on Vectorized
    Embedding

    Args:
        all_embeddings ([List]): [List of the embedding]

    """
    n_clusters = 3
    cmap = plt.get_cmap('Set2')
    clus = KMeans(n_clusters=n_clusters,
                  random_state=42)
    clusters = clus.fit_predict(vectorized)
    plt.scatter(umap_emb[:, 0],
                umap_emb[:, 1],
                c=[cmap(x/n_clusters) for x in clusters],
                s=40, alpha=.4)
    plt.title('UMAP projection of Sentence, colored by clusters',
              fontsize=14)


def word_cloud_viz(pos,
                   neg,
                   neutral,
                   preprocess=False):
    if preprocess:
        list_l = []
        for i in [pos, neg, neutral]:

            i_l = list(map(review_to_words, i))
            i_pre = list(map(lambda x: ' '.join([w for w in x]), i_l))
            list_l.append(i_pre)
        pos = list_l[0]
        neg = list_l[1]
        neutral = list_l[2]

    combined_text_pos = " ".join([r for r in pos])
    combined_text_neg = " ".join([r for r in neg])
    combined_text_neutral = " ".join([r for r in neutral])
    # Initialize wordcloud object
    wc = WordCloud(background_color='white',
                   max_words=50,
                   # update stopwords can be done
                   stopwords=STOPWORDS.update(['data'])
                   )
    fig, ax = plt.subplots(
        nrows=1, ncols=3, figsize=(16, 10)
    )
    axes = ax.ravel()
    # Generate and plot wordcloud
    axes[0].imshow(wc.generate(combined_text_pos))
    axes[0].set_title('Positive review')
    axes[1].imshow(wc.generate(combined_text_neg))
    axes[1].set_title('Negative review')
    axes[2].imshow(wc.generate(combined_text_neutral))
    axes[2].set_title('Neutral review')
    plt.axis('off')
    #plt.suptitle('differnt type of sentiments count')
    plt.tight_layout()
