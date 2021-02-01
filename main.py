import csv
import json
import sqlite3
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from stanfordcorenlp import StanfordCoreNLP
from textblob import TextBlob
import numpy as np
from numpy import vstack
from sklearn.metrics import cohen_kappa_score


def count_words(in_tweet):
    words = str.split(in_tweet)
    return len(words)


# check statements
def check_statements_txt_blob(in_txtblob):
    print(in_txtblob[0], in_txtblob[1])
    print("\n")
    print(in_txtblob)


def pol_nltk(in_score):
    temp = -1
    if in_score is not None:
        if abs(in_score['pos']) > abs(in_score['neg']):
            temp = 1
        elif abs(in_score['pos']) == abs(in_score['neg']):
            temp = 0
    return temp


def pol_nlp(in_score):
    temp = -1
    if int(in_score[1]) > 2:
        temp = 1
    elif int(in_score[1]) == 2:
        temp = 0
    return temp


def pol_blob(in_score):
    temp = 0
    if in_score is not None:
        if in_score[0] > 0:
            temp = 1
        elif in_score[0] < 0:
            temp = -1
    return temp


def ai_algorithm_comparison(score_nltk_f, score_nlp_f, score_txtblob_f):
    return abs((pol_nltk(score_nltk_f) + score_nlp_f + pol_blob(score_txtblob_f)) / 3)


# initializing  sentiment analysis objects

nlp = StanfordCoreNLP(r'C:\Users\salma\Downloads\code_transfers\stanford-corenlp-full-2018-10-05')
sentiment_analyser = SentimentIntensityAnalyzer()

all_scores_nltk = np.array([-10.0, -10.0, -10.0, -10.0])
all_scores_corenlp = np.array(["", 0])
all_scores_txtblob = np.array([0.0, 0.0])

skip_data_write = False  # skips csv read and db write
skip_variable_sent = False  # skips sentiment analysis when set to true...

comp_scores = np.array([-10, "tweet", "raters", "nltk", "corenlp", "text_blob", "comp"])

ck_nltk = np.array([])
ck_nlp = np.array([])
ck_blob = np.array([])
ck_rater = np.array([])

with open(r".\gold_rating_set.csv", 'r', encoding='utf-8-sig') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
        # Annotating tweets with sentiment scores
        score_nltk = sentiment_analyser.polarity_scores(row[1])
        score_corenlp = nlp.annotate(row[1], properties={'annotators': 'sentiment', 'outputFormat': 'json',
                                                         'timeout': 1000})
        score_txtblob = TextBlob(row[1]).sentiment
        # eof annotation code....

        # scores from stanford core nlp
        temp_score = ["", 0]
        try:
            # loading json
            temp_sentence_obj = json.loads(score_corenlp)
            # scores from stanford core nlp; splitting and copy values. (with error catching.)
            for s in temp_sentence_obj["sentences"]:
                try:
                    temp_score[0] = s["sentiment"]
                    temp_score[1] = s["sentimentValue"]
                except ValueError:
                    temp_score[0] = "empty"
                    temp_score[1] = -100
        except ValueError:
            temp_sentence_obj = {"sentences": {"sentiment": "empty", "sentimentValue": -100}}
            print("json err")

        check_score = pol_nlp(temp_score)

        # comp_scores = np.array([-10 ,"tweet", -10, -10, -10, -10])
        # task_data_1 = cohen_kappa_score(row[5], pol_nltk(score_nltk))
        # task_data_2 = cohen_kappa_score(row[5], check_score)
        # task_data_3 = cohen_kappa_score(row[5], pol_blob(score_txtblob))
        # print(row[5])
        # print(pol_nltk(score_nltk))

        ck_nltk_temp = np.array([pol_nltk(score_nltk)])
        ck_nlp_temp = np.array([check_score])
        ck_blob_temp = np.array([pol_blob(score_txtblob)])

        try:
            ck_rater_temp = np.array([int(row[5])])
            ck_rater = np.append(ck_rater, ck_rater_temp)
        except ValueError:
            print("first_error")

        ck_nltk = np.append(ck_nltk, ck_nltk_temp)
        ck_nlp = np.append(ck_nlp, ck_nlp_temp)
        ck_blob = np.append(ck_blob, ck_blob_temp)


        comp_scores_temp = np.array(
            [line_count, row[1], row[5], pol_nltk(score_nltk), check_score, pol_blob(score_txtblob),
             ai_algorithm_comparison(score_nltk, check_score, score_txtblob)])
        comp_scores = vstack((comp_scores, comp_scores_temp))

        # print(str(line_count) + "\n")
        line_count += 1

nlp.close()

# table for comparisons
csvFile = open(r".\comp_data_f4.csv", 'a', encoding='utf-8-sig')
csvWriter = csv.writer(csvFile)

for row in comp_scores:
    if row[1] != "":
        csvWriter.writerow(row)

csvFile.close()

# print(ck_rater)
df = pd.read_csv(r".\responses.csv")
df = df.dropna()
# print(df.head())
# print(df["Tweet_Ratings"])
# print(df["Gold Rating"])

df2 = pd.read_csv(r".\rater_comparison.csv")
# df2 = df2.dropna()
print(df2.head())

new_ck_rating = ck_rater
new_nltk_ratings = np.delete(ck_nltk, 0)
new_nlp_ratings = np.delete(ck_nlp, 0)
new_blob_ratings = np.delete(ck_blob, 0)

print("\n")
print(new_ck_rating)
print("\n")
print(ck_nltk)
print("\n")
print(ck_nlp)
print("\n")
print(ck_blob)
print("\n")

print(cohen_kappa_score(new_ck_rating, new_nltk_ratings))
print(cohen_kappa_score(new_ck_rating, new_nlp_ratings))
print(cohen_kappa_score(new_ck_rating, new_blob_ratings))
print(cohen_kappa_score(df["Tweet_Ratings"], df["Gold Rating"]))
