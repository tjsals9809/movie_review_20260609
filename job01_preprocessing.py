import pandas as pd
from konlpy.tag import Okt
import re

df = pd.read_csv('datasets/reviews_2017_2022.csv')
df.info()

df_stopwords = pd.read_csv('datasets/stopwords.csv')
stopwords = df_stopwords['stopword'].tolist()
stopwords = stopwords + ['가다', '감독', '연출', '연기', '배우', '보여주다', '좋다', '모르다', '주연', '많다', '들어가다',
                         '자다', '잃다', '현실', '부분', '오다', '채우다', '좋아하다', '나오다', '인간', '내다', '들다',
                         '처음', '정도', '영상', '버리다', '어떻다', ' 상황', '막다', '부르다', '만들다', '해주다', '싶다',
                         '떠나다', '받다', '진실', '모든', '진짜', '남다', '편의', '대단하다','엔딩', '늘다', '느껴지다',
                         '문제', '전개', '먹다', '끄다', '크다', '바람', '세기', '보이다', '알다', '남자', '이상하다',
                         '생기다', '작품', '안과', '원소', '높다', '알다', '전혀', '대사', '주다', '등장', '쏘다', '희생',
                         '출처', '이렇다','이루어지다', '덕분', '목격', '환경', '알파', '말씀', '아름답다', '살다' ]
okt = Okt()
print(df.titles[0])
print(df.reviews[0])

cleaned_sentences = []
for review in df.reviews:
    review = re.sub('[^가-힣]', ' ', review)
    tokened_review = okt.pos(review, stem = True)
    df_token = pd.DataFrame(tokened_review, columns = ['word', 'class'])
    df_token = df_token[(df_token['class'] == 'Noun') |
                        (df_token['class'] == 'Verb') |
                        (df_token['class'] == 'Adjective')]
    words = []
    for word in df_token['word']:
        if len(word) > 1:
            if word not in stopwords:
                words.append(word)
    cleaned_sentence = ' '.join(words)
    cleaned_sentences.append(cleaned_sentence)
df.reviews = cleaned_sentences
df.dropna(inplace = True)
df.info()
df.to_csv('./datasets/reviews_2017_2022_test.csv', index = False)






