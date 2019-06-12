import json
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)

def remove_stuff(t):
    return re.sub('(?:\\n|\\t\d?)', ' ', t)
def truncate(text):
    return text[0:min(len(text),1000)]

if __name__ == '__main__':
    data = fetch_20newsgroups(subset='train',
                              shuffle=True,
                              remove=('headers', 'footers', 'quotes'),
                              categories=['comp.windows.x', 'rec.sport.baseball', 'rec.motorcycles']
                              )
    target_names = data.target_names
    texts = [truncate(d) for d, target in zip(data.data, data.target) if len(d.split(' ')) > 5]
    print('num texts: %d'%len(texts))
    vectorizer = CountVectorizer(max_df=0.75, min_df=2,
                                 max_features=10000,
                                 tokenizer=lambda x: x.split(" ")
                                 )
    tf = vectorizer.fit_transform(texts)

    for n_topics in [10]:  # [20,40,80]
        lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=3, perp_tol=0.5,
                                          evaluate_every=3,
                                          verbose=True,
                                          learning_method='online',
                                          learning_offset=50.,
                                          random_state=0,
                                          n_jobs=-1)
        # t0 = time()
        lda.fit(tf)
        print("\nTopics in LDA model:")
        tf_feature_names = vectorizer.get_feature_names()
        print_top_words(lda, tf_feature_names, 9)
        # lda.transform(tf)
