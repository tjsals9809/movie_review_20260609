import pandas as pd
from gensim.models import Word2Vec

df_reviews = pd.read_csv('./datasets/reviews_2017_2022.csv')
df_reviews.info()

reviews = list(df_reviews.reviews)
print(reviews[0])

tokens = []
for sentence in reviews:
    token = sentence.split()
    tokens.append(token)
print(tokens[0])

# vertor_size: 의미벡터를 100개로 제한한다(단어 하나당 벡터 1개)
# min_count: 최소 20번 이상 등장하는 단어에 의미벡터를 부여하겠다
# workers: 코어의 수, epochs: 몇 번 반복할것인지,
embedding_model = Word2Vec(tokens, vector_size=100, window=4, min_count=20, workers=4, epochs=100, sg=1)
embedding_model.save('./models/word2vec_movie_review.model')
print(list(embedding_model.wv.index_to_key))
print(len(embedding_model.wv.index_to_key))







































