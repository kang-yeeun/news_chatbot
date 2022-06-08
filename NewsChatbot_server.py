import requests, json
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup as bs
from datetime import datetime
import pandas as pd
from pandas import DataFrame
from math import log
import gensim.downloader as api
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.summarization.summarizer import summarize#뉴스 속보
import re #조건부 문자열(정규 표현식), 태그 탐색 시 일반화 조건 사용 위함
import pyLDAvis
import pyLDAvis.gensim_models
import csv

from flask import Flask, render_template, request, jsonify

application = Flask(__name__)
@application.route('/Ranking')

def RankingNews():
    url="https://news.naver.com/main/ranking/popularDay.naver"
    headers={'User-Agent':'Mozilla/5.0'}

    res = requests.get(url, headers = headers)
    soup = BeautifulSoup(res.text , 'html.parser')

    all_box = soup.find_all('div', attrs={'class':'rankingnews_box'}) #신문사별 1~5위 데이터가 담긴 div 값 모두 가져옴

    all_data= [] #모든 데이터를 담을 리스트
    count = 1 #몇 번째 반복인지 알기위한 count 변수
    for box in all_box:
      press_name = box.strong.text #언론사 이름
      all_rank = box.find_all('li') #1~5위의 li값 가져옴
      rank_num = 1 #뉴스 랭킹 1위부터 입력
      for rank in all_rank :
        data_list = []
        data_list.append(datetime.now().strftime("%Y-%m-%d")) #오늘 날짜 "2022-04-24"형식으로 data_list에 추가
        data_list.append(press_name) #언론사 이름 추가
        if rank.a == None:
          continue #데이터가 없는 경우 다음 신문사로 넘어감
        data_list.append(rank_num) #뉴스 랭킹 추가
        data_list.append(rank.a.text) #뉴스 제목 추가
        data_list.append(rank.a["href"]) #뉴스 url 추가
        rank_num += 1
        all_data.append(data_list) #개별 신문사 순위 데이터를 all_data에 추가
  
  #print(f"전체 {str(len(all_box))} 중 {count}회 종료") #몇회째 실행 중인지
      count += 1

#데이터 프레임으로 변환
    df_1 = pd.DataFrame(all_data, columns=["날짜", "언론사", "순위", "제목", "url"])
    today = datetime.now().strftime("%Y%m%d")
    #df_1.to_csv(f"{today}RankingNews.csv", encoding = "utf-8-sig")해당 날짜를 제목에 더해서 저장
    df_1.to_csv(f"RankingNewsDF.csv", encoding = "utf-8-sig")
    
    df_1 = pd.read_csv(f"RankingNewsDF.csv")
    return df_1.to_html(header="true", table_id="table")#사이트에 데이터프레임 출력

#랭킹뉴스분석--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@application.route('/news')

def pyL_News():
    url="https://news.naver.com/main/ranking/popularDay.naver"
    headers={'User-Agent':'Mozilla/5.0'}

    res = requests.get(url, headers = headers)
    soup = BeautifulSoup(res.text , 'html.parser')

    all_box = soup.find_all('div', attrs={'class':'rankingnews_box'}) #신문사별 1~5위 데이터가 담긴 div 값 모두 가져옴

    all_data= [] #모든 데이터를 담을 리스트
    count = 1 #몇 번째 반복인지 알기위한 count 변수
    for box in all_box:
        press_name = box.strong.text #언론사 이름
        all_rank = box.find_all('li') #1~5위의 li값 가져옴
        rank_num = 1 #뉴스 랭킹 1위부터 입력
        for rank in all_rank :
            data_list = []
    #    data_list.append(datetime.now().strftime("%Y-%m-%d")) #오늘 날짜 "2022-04-24"형식으로 data_list에 추가
    #    data_list.append(press_name) #언론사 이름 추가
            if rank.a == None:
                continue #데이터가 없는 경우 다음 신문사로 넘어감
    #    data_list.append(rank_num) #뉴스 랭킹 추가
            data_list.append(rank.a.text) #뉴스 제목 추가
    #    data_list.append(rank.a["href"]) #뉴스 url 추가
            rank_num += 1
            all_data.append(data_list) #개별 신문사 순위 데이터를 all_data에 추가

      #print(f"전체 {str(len(all_box))} 중 {count}회 종료") #몇회째 실행 중인지
            count += 1

    #데이터 프레임으로 변환
    #df = pd.DataFrame(all_data, columns=["날짜", "언론사", "순위", "제목", "url"])
    #today = datetime.now().strftime("%Y%m%d") # 시간
    #df.to_csv(f"{today}RankingNews.csv", encoding = "utf-8-sig")

    all_data = sum(all_data, [])

    trans_data = []
    for i in all_data :
        table = str(all_data).maketrans({
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
            '盧' : '노무현',
            '北' : '북한',
            '英' : '영국',
            '前' : '',
            '故' : '',
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
        i = i.replace("한 ", " 한")
        i = i.replace(" 좀 ", " 좀")
        i = i.replace(" 는 ", " ")
        i = i.replace(" 해 ", " ")
        i = i.replace(" 된 ", " ")
        i = i.replace(" 걸 ", " ")
        i = i.replace(" 엄정 ", " ")
        i = i.replace(" 그 ", " ")
        i = i.replace(" 고 ", " ")
        i = i.replace(" 약 ", " ")
        i = i.replace(" 속 ", " ")
        i = i.replace(" 길 ", " ")
        i = i.replace(" 에 ", " ")
        i = i.replace("않는다", " ")
        i = i.replace("적당히", " ")
        i = i.replace("내가", " ")
        i = i.replace("to", " ")
        i = i.replace("not", " ")
        i = i.replace("by", " ")

        trans_data.append(i)

    df = pd.DataFrame(trans_data, columns = ["제목"])
    df.to_csv("RankingNews.csv", encoding = "utf-8-sig")


    # In[10]:


    token_list = [[text for text in doc.split()] for doc in trans_data]
    dct = Dictionary(token_list)

    term_matrix = [dct.doc2bow(token) for token in token_list]
    model = TfidfModel(term_matrix)

    dictionary = corpora.Dictionary(token_list)
    doc_term_matrix = [dictionary.doc2bow(tokens) for tokens in token_list]
    lda = LdaModel(corpus = doc_term_matrix, id2word = dictionary, num_topics = 10)
    lda.print_topics()
    vis = pyLDAvis.gensim_models.prepare(lda, doc_term_matrix, dictionary)
    pyLDAvis.save_html(vis, "./templates/LDAvis.html")

    return render_template('LDAvis.html')

#랭킹뉴스 워드클라우드-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@application.route('/WordCloud')
def WordCloud():
    url="https://news.naver.com/main/ranking/popularDay.naver"
    headers={'User-Agent':'Mozilla/5.0'}

    res = requests.get(url, headers = headers)
    soup = BeautifulSoup(res.text , 'html.parser')

    all_box = soup.find_all('div', attrs={'class':'rankingnews_box'}) #신문사별 1~5위 데이터가 담긴 div 값 모두 가져옴

    all_data= [] #모든 데이터를 담을 리스트
    count = 1 #몇 번째 반복인지 알기위한 count 변수
    for box in all_box:
        press_name = box.strong.text #언론사 이름
        all_rank = box.find_all('li') #1~5위의 li값 가져옴
        rank_num = 1 #뉴스 랭킹 1위부터 입력
        for rank in all_rank :
            data_list = []
    #    data_list.append(datetime.now().strftime("%Y-%m-%d")) #오늘 날짜 "2022-04-24"형식으로 data_list에 추가
    #    data_list.append(press_name) #언론사 이름 추가
            if rank.a == None:
                continue #데이터가 없는 경우 다음 신문사로 넘어감
    #    data_list.append(rank_num) #뉴스 랭킹 추가
            data_list.append(rank.a.text) #뉴스 제목 추가
    #    data_list.append(rank.a["href"]) #뉴스 url 추가
            rank_num += 1
            all_data.append(data_list) #개별 신문사 순위 데이터를 all_data에 추가

      #print(f"전체 {str(len(all_box))} 중 {count}회 종료") #몇회째 실행 중인지
            count += 1

    #데이터 프레임으로 변환
    #df = pd.DataFrame(all_data, columns=["날짜", "언론사", "순위", "제목", "url"])
    #today = datetime.now().strftime("%Y%m%d") # 시간
    #df.to_csv(f"{today}RankingNews.csv", encoding = "utf-8-sig")

    all_data = sum(all_data, []) # 2차원 리스트 -> 1차원 리스트

    trans_data = [] # 전처리한 데이터를 담을 리스트 (1차원)
    for i in all_data :
        table = str(all_data).maketrans({
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
            '盧' : '노무현',
            '北' : '북한',
            '英' : '영국',
            '前' : '',
            '故' : '',
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
        i = i.replace("한 ", " 한")
        i = i.replace(" 좀 ", " 좀")
        i = i.replace(" 는 ", " ")
        i = i.replace(" 해 ", " ")
        i = i.replace(" 된 ", " ")
        i = i.replace(" 걸 ", " ")
        i = i.replace(" 엄정 ", " ")
        i = i.replace(" 그 ", " ")
        i = i.replace(" 고 ", " ")
        i = i.replace(" 약 ", " ")
        i = i.replace(" 속 ", " ")
        i = i.replace(" 길 ", " ")
        i = i.replace(" 에 ", " ")
        i = i.replace("않는다", " ")
        i = i.replace("적당히", " ")
        i = i.replace("내가", " ")
        i = i.replace("to", " ")
        i = i.replace("not", " ")
        i = i.replace("by", " ")

        trans_data.append(i)

    df = pd.DataFrame(trans_data, columns = ["제목"])
    df.to_csv("RankingNews.csv", encoding = "utf-8-sig", index = False)


    # 전처리가 끝난 데이터를 이용해 토큰 생성
    token_list = [[text for text in doc.split()] for doc in trans_data]
    dictionary = corpora.Dictionary(token_list)
    doc_term_matrix = [dictionary.doc2bow(tokens) for tokens in token_list]

    lda = LdaModel(corpus = doc_term_matrix, id2word = dictionary, num_topics = 14)

    # 토픽별 유사도 계산


    # dictionary.token2id(dict 타입)의 키 <-> 밸류 뒤집음 (키가 index, 밸류가 단어가 되게 함)
    value_to_key = {v:k for k, v in dictionary.token2id.items()}

    term_matrix = [dictionary.doc2bow(token) for token in token_list]
    #model = TfidfModel(term_matrix)

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



    # In[4]:


    for i in range(1,topic_num+1):
        globals()['df_{}'.format(i)]=df_topic.loc[df_topic.Dominant_Topic==str(i)]
        globals()['df_{}'.format(i)].sort_values('Topic_Perc_Contrib',ascending=False,inplace = True)



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


    freq=pd.Series(data_word).value_counts().head(80)
    freq=dict(freq)

    wordcloud = WordCloud(font_path="./Binggrae-Bold.ttf",
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

    plt.savefig("./topic("+str(i)+")wordcloud.png")

#뉴스속보--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
application.route('/recent')

def Recent(): 
    start = 1
    result_df = pd.DataFrame()

    while start < 30:
      try :
        url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=%5B%EC%86%8D%EB%B3%B4%5D&sort=0&photo=0&field=0&pd=0&ds=&de=&cluster_rank=141&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:r,p:all,a:all&start=" + str(start)
        headers = {'user-agent':'Mozilla/5.0 '}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        news_title = [title['title'] for title in soup.find_all('a', attrs={'class':'news_tit'})] # 기사 제목
        news_url = [ url['href'] for url in soup.find_all('a', attrs={'class':'info'}) ] # 기사 url
        news_url = [i for i in news_url if 'news.naver.com' in i]
        dates = [ date.get_text() for date in soup.find_all('span', attrs={'class':'info'})] # 기사 작성일

        df = pd.DataFrame({'기사작성일':dates,'기사제목':news_title,'기사주소':news_url})
        result_df = pd.concat([result_df, df], ignore_index=True)


        start += 10

      except:
        print(start)
        break

    #본문
    contents = []

    for i in range(30):
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

    for i in range(30):
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

    result_df = pd.DataFrame(news_info, columns = ['시간', '제목', 'url', '본문'])
    result_df.to_csv("recentnews.csv", encoding = "utf-8-sig")
    result_df = pd.read_csv("recentnews.csv")
    return result_df.to_html(header="true", table_id="table")#사이트에 데이터프레임 출력

#구글 트랜드, 네이트, 줌 실시간 검색 순위------------------------------------------------------------------------------------------------------------------------------------------------------------
application.route('/trends')

def trends():
    now = datetime.now().strftime('%Y%m%d%H%M')
    url = 'https://www.nate.com/js/data/jsonLiveKeywordDataV1.js?v=' + now
    r = requests.get(url).content
    keyword_list = json.loads(r.decode('euc-kr'))
    nate_result = []
    for k in keyword_list:
        nate_result.append(k[1])
    with open("nate_trends.csv", 'w') as file: 
      writer = csv.writer(file) 
      writer.writerow(nate_result)

    #줌 실시간 검색어 크롤링
    url = 'https://m.search.zum.com/search.zum?method=uni&option=accu&qm=f_typing.top&query='
    html = requests.get(url).content
    soup = bs(html, 'html5lib')
    keyword_list = soup.find('div', {'class' : 'list_wrap animate'}).find_all('span', {'class' : 'keyword'})
    zum_result = []
    for k in keyword_list:
        zum_result.append(k.text.strip())

    with open("zum_trends.csv", 'w') as file: 
      writer = csv.writer(file) 
      writer.writerow(zum_result)



    #구글 트랜드 일간 인기 검색어 가져오기
    from pytrends.request import TrendReq
    import pandas as pd
    pytrends = TrendReq(hl="ko", tz=360)
    pytrends = pytrends.trending_searches(pn='south_korea')
    pytrends.to_csv("google_trends.csv", mode = 'w', index = None)

    zum_df = pd.read_csv("zum_trends.csv")
    zum_df = zum_df.transpose()
    zum_df.to_csv("zum_trends.csv")
    zum_df = pd.read_csv("zum_trends.csv")
    zum_df = zum_df.reset_index()
    zum_df

    nate_df = pd.read_csv("nate_trends.csv")
    nate_df = nate_df.transpose()
    nate_df.to_csv("nate_trends.csv")
    nate_df = pd.read_csv("nate_trends.csv")
    nate_df = nate_df.reset_index()
    nate_df


#카톡 챗봇 응답--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@application.route('/',methods=['POST'])
def reply():
    req = request.get_json()
    news_type = req["action"]["detailParams"]["News_type"]["value"]	# json파일 읽기
    answer = news_type
    
    #기능 명령어
    if answer == "명령어":
        res = {
          "version": "2.0",
          "template": {
            "outputs": [
              {
                "basicCard": {
                  "title": "챗봇 도움말",
                  "description": "실시간 뉴스 알림이 명령어 목록입니다!\n아래 명령어 버튼을 누르거나 해당 명령어를 입력하면 그에 맞는 기능이 실행됩니다!\n또 특정 언론사를 입력하시면 해당 언론사의 랭킹뉴스 5개의 헤드라인이 출력됩니다! ",
                  "buttons": [
                    {
                      "action":  "message",
                      "label": "랭킹뉴스분석",
                      "messageText": "랭킹뉴스분석"
                    },
                      
                    {
                      "action":  "message",
                      "label": "실시간 인기 검색어",
                      "messageText": "실시간 인기 검색어"
                    },
                      
                    {
                      "action":  "message",
                      "label": "속보",
                      "messageText": "속보"
                    }, 
                      
                  ]
                }
              }
            ]
          }
        }
    elif answer == "언론사별 랭킹뉴스":   
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText":
                        {
                            "text": "원하는 언론사를 입력해보세요! 해당 언론사의 랭킹뉴스 5개의 헤드라인이 출력됩니다!"
                        }
                    }
                ]
            }
        }    
    
    #랭킹뉴스분석 링크 주는 응답@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    elif answer == "랭킹뉴스분석":    
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        
                        "simpleText": 
                        {
                            "text": "https://rankingnews.run.goorm.io/news"
                        }

                    },
                    {
                        "simpleText":
                        {
                            "text": "뉴스를 분석하는데 시간이 걸려서 로딩이 많이 걸립니다! 양해 부탁드립니다!"
                        }
                    }
                ]
            }
        }
        print ("lalala") #카카오톡 응답 확인
    
    #구글 트렌드, 줌, 네이트 키워드 순위 응답@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    elif answer =="실시간 인기 검색어":
        res = {
          "version": "2.0",
          "template": {
            "outputs": [
              {
                "basicCard": {
                  "title": "실시간 인기 검색어",
                  "description": "세 사이트에서 실시간으로 가장 많이 검색되는 인기 키워드를 알려드립니다!",
                  "buttons": [
                    {
                      "action":  "message",
                      "label": "구글 검색어 트랜드",
                      "messageText": "구글 검색어 트랜드"
                    },
                      
                    {
                      "action":  "message",
                      "label": "Zum 인기 검색어",
                      "messageText": "Zum 인기 검색어"
                    },
                      
                    {
                      "action":  "message",
                      "label": "Nate 인기 검색어",
                      "messageText": "Nate 인기 검색어"
                    }
                  ]
                }
              }
            ]
          }
        }
    elif answer ==  "구글 검색어 트랜드":
        trends()
        df_4 = pd.read_csv("google_trends.csv")
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
	                    {
                        "listCard": {
                              "header": {
                                "title": "구글 트랜드 검색어 TOP 5"
                              },
                        "items": [
                            {
                                "title": df_4.iloc[0, 0]                             
                            },
                            {
                                "title": df_4.iloc[1, 0]                              
                            },
                            {
                                "title": df_4.iloc[2, 0]                              
                            },
                            {
                                "title": df_4.iloc[3, 0]                            
                            },
                            {
                                "title": df_4.iloc[4, 0]                            
                            }

                            ]
                        }
                    }
                ]
            }
        }
        
    elif answer ==  "Zum 인기 검색어":
        trends()
        df_5 = pd.read_csv("zum_trends.csv")
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
	                    {
                        "listCard": {
                              "header": {
                                "title": "Zum 인기 검색어 TOP 5"
                              },
                        "items": [
                            {
                                "title": df_5.iloc[0, 0]                             
                            },
                            {
                                "title": df_5.iloc[1, 0]                             
                            },
                            {
                                "title": df_5.iloc[2, 0]                             
                            },
                            {
                                "title": df_5.iloc[3, 0]                         
                            },
                            {
                                "title": df_5.iloc[4, 0]                           
                            }

                            ]
                        }
                    }
                ]
            }
        }
        
    elif answer ==  "Nate 인기 검색어":
        trends()
        df_5 = pd.read_csv("nate_trends.csv")
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
	                    {
                        "listCard": {
                              "header": {
                                "title": "Nate 인기 검색어 TOP 5"
                              },
                        "items": [
                            {
                                "title": df_5.iloc[0, 0]                             
                            },
                            {
                                "title": df_5.iloc[1, 0]                             
                            },
                            {
                                "title": df_5.iloc[2, 0]                             
                            },
                            {
                                "title": df_5.iloc[3, 0]                         
                            },
                            {
                                "title": df_5.iloc[4, 0]                           
                            }

                            ]
                        }
                    }
                ]
            }
        }
        
    #속보 알려주는 응답@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    elif answer == "속보":

        df_3 = pd.read_csv("newsflash.csv")
        df_3

        res = {
  "version": "2.0",
  "template": {
    "outputs": [
      {
            "simpleText":
            {
                "text": "아래에 출력되는 속보들은 관련도순으로 나열되었습니다! 이 점 참고해주세요!"
            }
      },
      {
        "carousel": {
          "type": "basicCard",
          "items": [
            {
              "title": df_3.iloc[0, 1],
              "description": "(" + df_3.iloc[0, 0] + ") " + df_3.iloc[0, 2],
              "thumbnail": {
                "imageUrl": "https://imgnews.pstatic.net/image/025/2022/06/08/0003200801_001_20220608130901130.jpg?type=w647"
              },
              "buttons": [
                {
                  "action":  "webLink",
                  "label": "기사 전문 보러가기",
                  "webLinkUrl": df_3.iloc[0, 3]
                }
              ]
            },
            {
              "title": df_3.iloc[1, 1],
              "description": "(" + df_3.iloc[1, 0] + ") " + df_3.iloc[1, 2],
              "thumbnail": {
                "imageUrl": "https://imgnews.pstatic.net/image/015/2022/06/08/0004708892_001_20220608132001092.jpg?type=w647"
              },
              "buttons": [
                {
                  "action":  "webLink",
                  "label": "기사 전문 보러가기",
                  "webLinkUrl": df_3.iloc[1, 3]
                }
              ]
            },
            {
              "title": df_3.iloc[2, 1],
              "description": "(" + df_3.iloc[2, 0] + ") " + df_3.iloc[2, 2],
              "thumbnail": {
                "imageUrl": "https://imgnews.pstatic.net/image/003/2022/06/08/NISI20181228_0014762897_web_20181228152340_20220608095107024.jpg?type=w647"
              },
              "buttons": [
                {
                  "action":  "webLink",
                  "label": "기사 전문 보러가기",
                  "webLinkUrl": df_3.iloc[2, 3]
                }
              ]
            }
          ]
        }
      }
    ]
  }
}
        print ("muyaho") #카카오톡 응답 확인
        
    #입력된 언론사별 랭킹뉴스 헤드라인 응답@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    else:
        RankingNews()
        df_2 = pd.read_csv(f"RankingNewsDF.csv")#랭킹뉴스
        df_2 = df_2[df_2['언론사'] == answer] #특정 언론사에 대한 랭킹뉴스만 추출
        df_2.to_csv(f"{answer}RankingNews.csv", encoding = "utf-8-sig")
        df_2
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "listCard": {
                              "header": {
                                "title": f"{answer} 랭킹뉴스 헤드라인입니다."
                              },
                        "items": [
                            {
                              "title": df_2.iloc[0, 4],

                              "link": {
                                "web": df_2.iloc[0, 5]
                              }
                            },
                            {
                              "title": df_2.iloc[1, 4],

                              "link": {
                                "web": df_2.iloc[1, 5]
                              }
                            },
                            {
                              "title": df_2.iloc[2, 4],

                              "link": {
                                "web": df_2.iloc[2, 5]
                              }
                            },
                            {
                              "title": df_2.iloc[3, 4],

                              "link": {
                                "web": df_2.iloc[3, 5]
                              }
                            },
                            {
                              "title": df_2.iloc[4, 4],

                              "link": {
                                "web": df_2.iloc[4, 5]
                              }
                            },

                        ]
                        }
                    }
                ]
            }
        }
      
    return jsonify(res)


if __name__ == '__main__':
    # threaded=True 로 넘기면 multiple plot이 가능해짐
  application.run(host="0.0.0.0",port=5000,threaded=True) 