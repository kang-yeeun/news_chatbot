# -*- coding: utf-8 -*-
"""속보 크롤링.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DtTHyVsnVlE6xcc9pBWNBuLqSgDXAImO
"""

!pip install konlpy

"""# 속보 크롤링"""

from bs4 import BeautifulSoup #HTML 파싱, 필요 태그 및 소스 추출
import requests # 웹 페이지 소스 추출(HTML)
import re #조건부 문자열(정규 표현식), 태그 탐색 시 일반화 조건 사용 위함
import pandas as pd
from gensim.summarization.summarizer import summarize #뉴스 요약

start = 1
result_df = pd.DataFrame()

start = 1
result_df = pd.DataFrame()
while start < 50:
    try :
        url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=%5B%EC%86%8D%EB%B3%B4%5D&sort=0&photo=0&field=0&pd=0&ds=&de=&cluster_rank=141&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:r,p:all,a:all&start=" + str(start)
        headers = {'user-agent':'Mozilla/5.0 '}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        news_title = []
        news_url = []
        dates = []
        
        naver = soup.find_all('a', attrs = {'class' : 'info'})
        for i in naver :
            if 'news.naver.com' in i['href'] :
                news_title.append(i.parent.parent.next_sibling['title'])
                news_url.append(i['href'])
                dates.append(i.previous_sibling.get_text())
    
        df = pd.DataFrame({'기사작성일':dates,'기사제목':news_title,'기사주소':news_url})
        result_df = pd.concat([result_df, df], ignore_index=True)

   
        start += 10
    
    except Exception as e :
        print(e)
        break

#본문
contents = []

for i in range(len(result_df)):
  news_r = requests.get(result_df['기사주소'][i], headers = headers) 
  news_sp = BeautifulSoup(news_r.text, 'html.parser')
  content= news_sp.find('div', 'go_trans _article_content')
  content = str(content)
  content = re.sub('<.+?>','',content,0).strip()
  content = re.sub(r'\[[^\]]*\]|\([^\)]*\)|\{[^\}]*\}' ,'',content,0).strip() #괄호 안 문자열 제거
  content = re.sub(r'\n*|\t*|\r\n*' ,'',content,0).strip() #줄바꿈 공백 제거
  content = re.sub(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)','',content,0).strip() #이메일 제거

  contents.append(content)
result_df['본문'] = contents  

result_df

"""# 본문 요약(Gensim 라이브러리 사용)"""

#본문 요약
news_info = result_df.values.tolist()
news_info

for i in range(len(result_df)):
  try:
    snews_contents = summarize(news_info[i][3], word_count = 20)
  except:
    snews_contents  = None

  if not snews_contents:
    news_sentences = news_info[i][3].split('.')
    if len(news_sentences) > 3:
      snews_contents = '.'.join(news_sentences[:3])
    else:
      snews_contents = '.'.join(news_sentences)
  news_info[i][3] = snews_contents

result_df = pd.DataFrame(news_info, columns = ['시간', '제목', 'url', '본문'])

for i in range(len(result_df)):
  if len(result_df['본문'][i]) < 10:
    result_df = result_df.drop([i])
result_df = result_df.reset_index(drop=True)
result_df

"""# TF-IDF 및 코사인 유사도"""

import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt  
okt=Okt()

def tokenizer(raw, pos=["Noun","Alpha","Number"], stopword=[]):
    return [
        word for word, tag in okt.pos(
            raw, 
            norm=True,   
            stem=True    
            )
            if len(word) > 1 and tag in pos and word not in stopword
        ]
df_title = []

for i in range(len(result_df)):
  df_title.append(result_df['제목'][i])

vectorizer = TfidfVectorizer(tokenizer = tokenizer, min_df = 2)

tfidf_data = vectorizer.fit_transform(df_title)
print()

tf_idf_df = pd.DataFrame(tfidf_data.toarray())

cos_sim_df = pd.DataFrame(cosine_similarity(tf_idf_df, tf_idf_df))
cos_sim_df

"""# 중복기사 제거"""

similar_val = []
for i in range(len(cos_sim_df)-1) : 
    if (max(cos_sim_df[i][(i+1):])) < 0.3 :
        similar_val.append(i)
similar_val
time = []
title = []
txt = []
url = []

for i in similar_val:
  time.append(result_df['시간'][i])
  title.append(result_df['제목'][i])
  txt.append(result_df['본문'][i])
  url.append(result_df['url'][i])

newsflash = df = pd.DataFrame({'작성 시간':time,'제목':title,'본문 내용':txt, 'url' : url})
newsflash.to_csv("newsflash.csv",index=False, encoding="utf-8-sig")
newsflash

