import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('movie_reviews')
nltk.download('punkt')

import string
from random import shuffle
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier
from nltk import classify
from nltk import ngrams
from nltk.tokenize import word_tokenize
import requests
import csv
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, tree
from os import path

file = "dataRate.csv"
def crawl(ymin, ymax, end):
  stop_eng = stopwords.words('english')
  def clean_words(words, stop_eng): #remove stopwords and punctuation
    clean = []
    for word in words:
      word = word.lower()
      if word not in stop_eng and word not in string.punctuation:
        clean.append(word)
    return clean

  def bag_words(words): #feature extractor for unigram
    dictionary = dict([word, True] for word in words)
    return dictionary
  def bag_ngrams(words, n=2):
    ng = []
    for item in iter(ngrams(words, n)):
      ng.append(item)
    dictionary = dict([word, True] for word in ng)
    return dictionary

  important = ['above', 'below', 'off', 'over', 'under', 'more', 'most', 'such', 'no', 'nor', 'not', 'only', 'so', 'than', 'too', 'very', 'just', 'but']
  stop_bi = set(stop_eng) - set(important)
  def bag_of_all(words, n=2):
    clean = clean_words(words, stop_eng)
    clean_bi = clean_words(words, stop_bi)
    feature_bi = bag_ngrams(clean_bi)
    feature = bag_words(clean)
    feature.update(feature_bi)
    return feature

  pos_set = []
  for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_set.append((bag_of_all(words), 'pos'))

  neg_set = []
  for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_set.append((bag_of_all(words), 'neg'))
  
  shuffle(pos_set)
  shuffle(neg_set)

  test = pos_set[:200] + neg_set[:200]
  training = pos_set[200:] + neg_set[200:]
  classifier = NaiveBayesClassifier.train(training)

  with open(file, 'a+', newline='') as fp:
    w = csv.writer(fp)
    
    for i in range(0, end):
      print("Loading...")
      
      url = 'https://www.imdb.com/search/title/?title_type=feature&release_date= %s-01-01,%s-12-31&groups=oscar_nominee&start=%s&ref_=adv_nxt' %(ymin, ymax,(1+i*50))
      web = requests.get(url)
      soup = BeautifulSoup(web.content, 'lxml')
      u = soup.find_all('div', class_ = 'lister-item mode-advanced')
      
      for url in u:
        title = url.h3.a.text.strip()
        link = url.h3.a.get('href')
        year = url.h3.find('span', class_ = 'lister-item-year text-muted unbold').text.strip()
        genre = url.p.find('span', class_ = 'genre').text.strip()
        rating = url.find('div', class_ = 'ratings-bar').div.text.strip()

        x = url.find('div', class_ = 'ratings-bar')
        if x.find('div', class_ = 'inline-block ratings-metascore'):
          metascore = int(x.find('div', class_ = 'inline- block ratings-metascore').span.text.strip())/10
        else:
          metascore = 0

        web2 = requests.get('https://www.imdb.com/%s'%(link))
        soup2 = BeautifulSoup(web2.content, 'lxml')
        x = soup2.find('div', id = 'titleDetails')
        if ("Country" in x.find_all('div', class_ = 'txt-block')[1].text):
          y = x.find_all('div', class_ = 'txt-block')[2]
        else:
          y = x.find_all('div', class_ = 'txt-block')[1]
        lan = y.text.strip().replace('Language:\n','').replace('\n|\n',', ')

        r_link = requests.get('https://www.imdb.com/%s/reviews?spoiler=hide&sort=help fulnessScore&dir=desc&ratingFilter=0' %(link[:16]))
        soup_r = BeautifulSoup(r_link.content, 'lxml')
        x = soup_r.find_all('div', class_ = 'lister-item-content')
        count = 0
        rate = 0
        pos = 0
        neg = 0
        result = ""
        for j in x:
          try:
            r = j.find('div', class_ = 'ipl-ratings-bar').span.text.strip()
            cm = j.find('div', class_ = 'text show-more__control').text.strip()
            count = count + 1
            ur = r[:(r.index('/'))]
            rate += int(ur)
            tokens = word_tokenize(cm.lower())
            review = bag_of_all(tokens)
            prob_result = classifier.prob_classify(review)
            pos = pos + prob_result.prob("pos")
            neg = neg + prob_result.prob("neg")
          except:
            continue
        if count > 0:
          rate = rate / count
          cmprob = (pos - neg) / count
          if pos > neg:
            result = "pos"
          else:
            result = "neg"
          item = ([title, year, genre, rating, metascore, lan, rate, cmprob, result])
          w.writerow(item)
          
def yr_check():
  df = pd.read_csv(file, sep = ",")
  df['Year'] = df['Year'].apply(lambda x:x.replace('(','').replace(')','').replace('I','').replace(' ','').replace('V',''))
  df.drop_duplicates(subset='Title', keep='first', inplace=True) l = df.groupby('Year').groups.keys()
  list_y = []
  for yr in l:
    if "2" in yr:
      list_y.append(yr)
    ymin = min(list_y)
    ymax = max(list_y)
    return ymin, ymax

if path.exists(file):
  ymin, ymax = yr_check()
else:
  print("Data is not available. Crawling basic data...")
  with open(file, 'w+', newline='') as fp:
    w = csv.writer(fp)
    w.writerow(["Title", "Year", "Genre", "Rating", "Metascore/10", "Language", "User Rate", "Comment Score", "Result"])
    crawl(2008, 2015, 10)
    ymin = 2008
    ymax = 2015
    
def best_picture_list(year):
  item = []
  url = 'https://www.oscars.org/oscars/ceremonies/%s' %(year) web = requests.get(url)
  soup = BeautifulSoup(web.content, 'lxml')
  if year == "2011" or year == "2012":
    x = soup.find_all('div', class_ = 'view-grouping')[17]
  elif year == "2013":
    x = soup.find_all('div', class_ = 'view-grouping')[16]
  else:
    x = soup.find_all('div', class_ = 'view-grouping')[15]
  y = x.find_all('h4')
  for i in y:
    item.append(i.text)
  return item
  
def predict_best_picture(s_year, nominees):
  df = pd.read_csv(file, sep = ",")
  df['Year'] = df['Year'].apply(lambda x:x.replace('(','').replace(')','').replace('I','').replace(' ','').replace('V',''))
  df.drop_duplicates(subset=None, keep='first', inplace=True)
  
  df['User Rate'] = pd.to_numeric(df['User Rate']) + pd.to_numeric(df['Comment Score'])
  df['Evaluate'] = df['Metascore/10']*0.6+df['Rating']*0.2+df['User Rate']*0.2
  year_list = []
  for j in range(2008, int(s_year)):
    year_list.append(j)

  yr = '|'.join(str(elem) for elem in year_list)
  df0 = df[df['Year'].str.contains(yr)].copy()
  NameList = ['Slumdog Millionaire', 'WALL-E', 'The Hurt Locker', 'Up', 'The King\'s Speech', 'Toy Story 3', 'The Artist', 'Rango','Argo','Brave','12 Years a Slave','Frozen','Birdman or (The Unexpected Virtue of Ignorance)','Big Hero 6', 'Spotlight', 'Inside out','Moonlight','Zootopia','The Shape of Water', 'Coco', 'Green Book', 'Spider-Man: Into the Spider-Verse']

  def finder(x):
    if x in NameList:
      return 1
    else:
      return 0

  df0['Award'] = df0.Title.apply(lambda x: finder(x))
  columns = ['Rating', 'Evaluate', 'Award']
  columns2 = ['Rating', 'Evaluate', 'Title']
  past = df0[columns]
  df1 = df[df['Year'].str.contains(str(s_year))].copy()
  
  def nominee_df(x):
    if x in nominees:
      return 1
    else:
      return 0

  df1['Nominee'] = df1.Title.apply(lambda x: nominee_df(x))
  dfN = df1[df1['Nominee'] == 1]
  future = dfN[columns2]

  films = future['Title']
  features = future[['Rating', 'Evaluate']]
  predt = {}
  for f in films:
    predt[f] = 0.0
  X = past[['Evaluate']]
  Y = past[['Rating']]

  Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y.values.ravel(), test_size = 0.6, random_state = 1)
  
  dtree = DecisionTreeClassifier()
  enc = preprocessing.LabelEncoder()
  encoded = enc.fit_transform(Ytrain)
  dtree.fit(Xtrain, encoded)

  for j, f in enumerate(films):
    predt[f] = float(dtree.predict(features[j:j+1][['Evaluate']]))
  
  for key, value in sorted(predt.items(), key=lambda p:p[1], reverse=True)[:3]:
    print(key, " : ", value)

state = True
while state:
  ymin, ymax = yr_check()
  print("\nYour database is now containing data between %s and %s." %(ymin, ymax))
  choice = input ("Please input the year to be evaluated (0 to exit or 1 to have more information on the statistic): ")
  try:
    val = int(choice)
    if val >= 2010 and val <= 2020:
      if val <= int(ymax):
        print("All data is in the database. Now is predicting The Best Picture of %s" %(choice))
        nominees = best_picture_list(str(val+1))
        predict_best_picture(val, nominees)
      elif val > int(ymax):
        print("Data is not available. Crawling data...")
      if val - int(ymax) == 1:
        crawl(choice, choice, 2)
      else:
        crawl(str(int(ymax)+1), choice, (val-int(ymax))*2)

      nominees = best_picture_list(str(val+1))
      predict_best_picture(val, nominees)
    elif val == 0:
      state = False
      print("Thank you for using.")
    elif val == 1:
      choice = input ("Please input the year to be investigated: ")
      try :
        val = int(choice)
        if val >= 2008 and val <= int(ymax):
          df = pd.read_csv(file, sep = ",")
          df['Year'] = df['Year'].apply(lambda x: x.replace('(','').replace(')','').replace('I','').replace('','').replace('V',''))
          df.drop_duplicates(subset=None, keep='first', inplace=True)
          df['User Rate'] = pd.to_numeric(df['User Rate']) + pd.to_numeric(df['Comment Score'])
          df0 = df[df['Year'].str.contains(choice)].copy() nominees = best_picture_list(str(int(choice)+1))
          def nominee_df(x):
            if x in nominees:
              return 1
            else:
              return 0
              
          df0['Nominee'] = df0.Title.apply(lambda x:nominee_df(x))
          dfN = df0[df0['Nominee'] == 1]
          print(dfN[['Title', 'Genre', 'Language', 'Rating', 'Metascore/10', 'User Rate']])
        else:
          print("Data input not correct. Please try again.")
      except ValueError:
        print("Data input not correct. Please try again.")
    else:
        print("Please input year between 2010 and 2020")
  except ValueError:
    print("Please input integer value between 2010 and 2020")
