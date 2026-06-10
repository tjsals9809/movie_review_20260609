import pandas as pd
from scipy.interpolate import dfitpack
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import mmwrite, mmread # matrix를 읽고 쓰는 함수
import pickle

df_reviews = pd.read_csv('./datasets/reviews_2017_2022.csv')
df_reviews.info()

Tfidf = TfidfVectorizer(sublinear_tf = True)
Tfidf_matrix = Tfidf.fit_transform(df_reviews.reviews)
print(Tfidf_matrix.shape)

with open('./models/tfidf.pkl', 'wb') as f:
    pickle.dump(Tfidf, f)
mmwrite('./models/Tfidf_movie_review.mtx', Tfidf_matrix)