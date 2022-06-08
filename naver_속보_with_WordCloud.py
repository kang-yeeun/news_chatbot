#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup #HTML 파싱, 필요 태그 및 소스 추출
import requests # 웹 페이지 소스 추출(HTML)
import re #조건부 문자열(정규 표현식), 태그 탐색 시 일반화 조건 사용 위함
import pandas as pd
from gensim.summarization.summarizer import summarize #뉴스 요약
from pyquery import PyQuery

start = 1
result_df = pd.DataFrame()

while start < 30:
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
    content = re.sub(r'\n*|\t*|\r\n*'  ,'',content,0).strip() #줄바꿈 공백 제거
    content = re.sub(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)','',content,0).strip() #이메일 제거

    contents.append(content)

result_df['본문'] = contents  

#본문 요약
news_info = result_df.values.tolist()
news_info

for i in range(len(result_df)):
    try:
        snews_contents = summarize(news_info[i][3], word_count = 30)
    except:
        snews_contents  = None

    if not snews_contents:
        news_sentences = news_info[i][3].split('.')
        if len(news_sentences) > 3:
            snews_contents = '.'.join(news_sentences[:3])
        else:
            snews_contents = '.'.join(news_sentences)
    news_info[i][3] = snews_contents
    

breaking_contents = pd.DataFrame(contents).to_csv('./BreakingNews.csv')

result_df = pd.DataFrame(news_info, columns = ['시간', '제목', 'url', '본문'])
result_df


# In[3]:


import pandas as pd
from math import log

data = pd.read_csv('./BreakingNews.csv').iloc[:, 1]
data = data.values.tolist()

trans_data = [] # 전처리한 데이터를 담을 리스트 (1차원)
for i in data :
    table = str(data).maketrans({
        '[' : ' ',
        '‘' : ' ',
        ']' : ' ',
        '’' : ' ',
        '"' : ' ',
        '\\' : ' ',
        '“' : ' ',
        '…' : ' ',
        '‥' : ' ',
        '.' : ' ',
        '”' : ' ',
        '·' : ' ',
        ',' : ' ',
        '\'' : ' ',
        '(' : ' ',
        ')' : ' ',
        '<' : ' ',
        '>' : ' ',
        '/' : ' ',
        '→' : ' ',
        '①' : ' ',
        '②' : ' ',
        '③' : ' ',
        '④' : ' ',
        '-' : ' ',
        'ㆍ' : ' ',
        '‧' : ' ',
        '｜' : ' ',
        '?' : ' ',
        '~' : ' ',
        '!' : ' ',
        '`' : ' ',
        'ㅣ' : ' ',
        '↔' : '',
        '檢' : '검찰',
        '美' : '미국',
        '尹' : '윤석열',
        '野' : '야당',
        '與' : '여당',
        '韓' : '한국',
        '靑' : '청와대',
        '日' : '일본',
        '獨' : '독일',
        '勢' : '세력',
        '株' : '주식',
        '軍' : '군대',
        '中' : '중국',
        '文' : '문재인',
        '安' : '안철수',
        '盧' : '노무현',
        '北' : '북한',
        '前' : '',
        '佛' : '프랑스'
    })
    i = i.translate(table)
    i = i.replace("단독","")
    i = i.replace("르포","")
    i = i.replace("속보", "")
    i = i.replace("첫 ", "첫")
    i = i.replace(" 것", "것")
    i = i.replace("vs", "")
    i = i.replace("5 18", " 오일팔 ")
    i = i.replace("518", " 오일팔 ")
    i = i.replace("우크라", " 우크라이나 ")
    i = i.replace(" 사단", "사단")
    i = i.replace("더 ", "")
    i = i.replace("새 ", "새")
    i = i.replace(" 보니", "")
    i = i.replace("영상 ", "")
    i = i.replace(" 영상", "영상")
    i = i.replace(" 있다", "있다")
    i = i.replace("대검 ", "대검")
    i = i.replace("버려진 ", "버려진")
    i = i.replace("잘 ", "잘")
    i = i.replace(" 공개", "공개")
    i = i.replace(" 등 ", " ")
    i = i.replace("Pick", "")
    i = i.replace("선 ", "")
    i = i.replace("윤 ", "윤석열")
    i = i.replace("뉴스쏙:속", "")
    i = i.replace("전 ", "전")
    i = i.replace("안 ", "안")
    i = i.replace("왜 ", "왜")
    i = i.replace(" 이 ", "")
    i = i.replace(" 만에", "만에")
    i = i.replace("문 대통령", "문재인 대통령")
    i = i.replace(" 러 ", " 러시아 ")
    i = i.replace("날 듯", "날듯")
    i = i.replace("나도 ", "나도")
    i = i.replace("최대 ", "최대")
    i = i.replace(" 또 ", " 또")
    i = i.replace(" 중 ", "중 ")
    i = i.replace(" 한 ", " 한")
    i = i.replace(" 좀 ", " 좀")
    i = i.replace(" 는 ", "")
    i = i.replace(" 해 ", "")
    i = i.replace(" 된 ", "")
    i = i.replace(" 고 ", "")
    i = i.replace(" 걸 ", "")
    i = i.replace(" 며 ", "")
    i = i.replace("않는다", "")
    i = i.replace("했다", "")
    i = i.replace("였다", "")
    i = i.replace("적당히", "")
    i = i.replace("내가", "")
    i = i.replace("to", "")
    i = i.replace("not", "")
    i = i.replace("by", "")
    
    trans_data.append(i)
    
df = pd.DataFrame(trans_data)
df.to_csv("BreakingNews_trans.csv", encoding = "utf-8-sig", index = False)


# In[8]:


import pyLDAvis
import pyLDAvis.gensim_models
from gensim.models.ldamodel import LdaModel

lda = LdaModel(corpus = doc_term_matrix, id2word = dictionary, num_topics = 10)
lda.print_topics()
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda, doc_term_matrix, dictionary)
vis


# In[9]:


# 토픽별 유사도 계산

import gensim.downloader as api
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

# 전처리가 끝난 데이터를 이용해 토큰 생성
token_list = [[text for text in doc.split()] for doc in trans_data]
dictionary = corpora.Dictionary(token_list)

# dictionary.token2id(dict 타입)의 키 <-> 밸류 뒤집음 (키가 index, 밸류가 단어가 되게 함)
value_to_key = {v:k for k, v in dictionary.token2id.items()}

term_matrix = [dictionary.doc2bow(token) for token in token_list]
#model = TfidfModel(term_matrix)

doc_term_matrix = [dictionary.doc2bow(tokens) for tokens in token_list]

coherence_model = CoherenceModel(model=lda, texts=token_list, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model.get_coherence()

# 유사도 계산 메소드
def compute_coherence_values(dictionary, corpus, texts, limit, start=4, step=2):

    coherence_values = [] # 토픽별 유사도 저장 리스트
    model_list = [] # 지정한 start, limit, step별로 만들 모델을 저장할 리스트
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=doc_term_matrix, texts=token_list, start=4, limit=21, step=2)


limit=21; start=4; step=2;
x = range(start, limit, step)
topic_num = 0
count = 0
max_coherence = 0
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", cv)
    coherence = cv
    if coherence >= max_coherence:
        max_coherence = coherence
        topic_num = m
        model_list_num = count   
    count = count+1
print("Number of Topics : ", topic_num)
        
# 유사도가 가장 높은 모델 사용
optimal_model = model_list[model_list_num]
model_topics = optimal_model.show_topics(formatted=False)


def format_topics_sentences(ldamodel=optimal_model, corpus=doc_term_matrix, texts=token_list):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    #ldamodel[corpus]: lda_model에 corpus를 넣어 각 토픽 당 확률을 알 수 있음
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num,topn=10)
                topic_keywords = ", ".join([value_to_key[int(word)] for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    #contents = pd.Series(texts)
    #sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=doc_term_matrix, texts=token_list)

# Format
df_topic = df_topic_sents_keywords.reset_index()
df_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords']


# Group top 5 sentences under each topic
sent_topics_sorteddf = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf = pd.concat([sent_topics_sorteddf, 
                                      grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                      axis=0)

# Reset Index    
sent_topics_sorteddf.reset_index(drop=False, inplace=True)


topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
topic_counts.sort_index(inplace=True)

topic_contribution = round(topic_counts/topic_counts.sum(), 4)
topic_contribution

lda_inform = pd.concat([sent_topics_sorteddf, topic_counts, topic_contribution], axis=1)
lda_inform.columns=["Index", "Topic_Num", "Topic_Perc_Contrib", "Keywords", "Num_Documents", "Perc_Documents"]
lda_inform = lda_inform[["Topic_Num","Keywords","Num_Documents","Perc_Documents"]]
lda_inform
#lda_inform.Topic_Num = lda_inform.Topic_Num.astype(int)
lda_inform['Topic_Num'] =lda_inform['Topic_Num'] +1
lda_inform.Topic_Num = lda_inform.Topic_Num.astype(str)
lda_inform['Topic_Num'] =lda_inform['Topic_Num'].str.split('.').str[0]
df_topic['Dominant_Topic'] =df_topic['Dominant_Topic'] +1
df_topic.Dominant_Topic = df_topic.Dominant_Topic.astype(str)
df_topic['Dominant_Topic'] =df_topic['Dominant_Topic'].str.split('.').str[0]

lda_inform.to_csv("lda_inform.csv", index = None)
lda_inform


# In[10]:


for i in range(1,topic_num+1):
    globals()['df_{}'.format(i)]=df_topic.loc[df_topic.Dominant_Topic==str(i)]
    globals()['df_{}'.format(i)].sort_values('Topic_Perc_Contrib',ascending=False,inplace = True)
    globals()['df_{}'.format(i)].to_csv ("./Result/topic("+str(i)+").csv", index = None)


# In[11]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

def flatten(l):
    flatList=[]
    for elem in l:
        if type(elem) == list:
            for e in elem:
                flatList.append(e)
        else:
            flatList.append(elem)
    return flatList

#토픽별 word cloud

data_word=[]

for i in range(1,topic_num+1):
    data_list = globals()['df_{}'.format(i)]['Keywords'].values
    
    for j in range(len(data_list)) :
        data_word.append(data_list[j].split(','))
    
    data_word = flatten(data_word)
    data_word=[x for x in data_word]
    
    
freq=pd.Series(data_word).value_counts().head(100)
freq=dict(freq)
    
wordcloud = WordCloud(font_path="./Font/Binggrae-Bold.ttf",
             relative_scaling = 0.2,
             background_color = 'white',
             max_font_size = 250,
             scale = 2.0,
             width = 400, height = 400
            ).generate_from_frequencies(freq)
plt.figure(figsize=(16,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

plt.savefig("./Result/topic("+str(i)+")wordcloud.png")

