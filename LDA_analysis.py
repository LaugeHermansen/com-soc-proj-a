#%%
from document_analysis_tools import (MyStemmer,
                                    Corpora,
                                    LDA,
                                    stopwords,
                                    open_data_files,
                                    tokenize_stem_remove)
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm

#%%


def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()



#%%

stemmer = MyStemmer("my_stemmer.pkl")
stemmed_stopwords = set(map(stemmer.stem, stopwords.words('english')))
data = open_data_files('comments', stemmer, stemmed_stopwords, restart = False)

# #%%

# # corpora_author = Corpora(data, 'author')
# #%%
# corpora_link_id = Corpora(data, 'link_id')
# # #%%
# # corpora_subreddit = Corpora(data, 'subreddit')

# #%%


# plt.hist(corpora_link_id.get_doc_size_distribution(), bins = 50)
# plt.yscale('log')




# #%%

# (feature_names,
# doc_names,
# doc_word_matrix_link_id) = corpora_link_id.get_doc_word_matrix(doc_min_size=50, recalculate=False)

# lda = LDA(n_components = 10, n_jobs = 6)

# lda.fit(doc_word_matrix_link_id)

# plot_top_words(lda, stemmer.reverse_stem(feature_names), 10, "Topics in LDA model")

# #%%

# topic_classification = lda.transform(doc_word_matrix_link_id)

# #%%

# tokens_dt_matrix = lil_matrix((len(data), len(feature_names)))

# for doc_id ,tokens in tqdm(enumerate(data.tokens)):
#     for token in tokens:
#         term_id = corpora_link_id.feature_map[token]
#         tokens_dt_matrix[doc_id, term_id] = 1

# tokens_dt_matrix = tokens_dt_matrix.tocsr()

# data['topic'] = lda.transform(tokens_dt_matrix)


#%%

corpora_topic = Corpora(data, 'topic')

#%%

for d in corpora_topic.get_documents():
    # plt.imshow(d[1].get_wordcloud(stemmer))
    # plt.title(d[0])
    # plt.show()
    print(d[1].get_wordcloud(stemmer))
