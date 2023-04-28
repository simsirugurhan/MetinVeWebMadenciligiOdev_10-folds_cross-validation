#Uğurhan Şimşir
import os
import re
import string

tweets = []  #Tweetleri depolamak için boş bir liste oluşur.
labels = []  #Etiketleri depolamak için boş bir liste oluşur.

#Verilen yoldaki tüm dizinlere ve alt dizinlere yürünüyor.
for root, dirs, files in os.walk('raw_texts'):
    #Mevcut dizindeki tüm dosyalara döngüyle erişiliyor.
    for file in files:
        #Dosya '.txt' ile bitiyor mu kontrol ediliyor.
        if file.endswith('.txt'):
            #Etiket olarak dizin adı alınıyor.
            labels.append(os.path.basename(root))
            #Dosyadan tweet okunuyor ve tweet_list'e ekleniyor.
            tweets.append((open(os.path.join(root, file), encoding="windows-1254").read().replace('\n', ' ').strip().lower()))

#Her sınıfa ait örnekleri ayır
class1 = [i for i in labels if i == '1']
class2 = [i for i in labels if i == '2']
class3 = [i for i in labels if i == '3']

#Uğurhan Şimşir

def preprocess_text(text):
    
    #Küçük harflere dönüştürme
    text = text.lower()
    
    #Kullanıcı adlarını kaldırma
    text = re.sub('@[^\s]+', '', text)

    #Hashtagleri kaldırma
    text = re.sub(r'#[^\s]+', '', text)

    #Linkleri kaldırma
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text)

    #Sayıları kaldırma
    text = ''.join(filter(lambda x: not x.isdigit(), text))

    #Noktalama işaretlerini kaldırma
    punctuation = string.punctuation + "”“‘’"  # Türkçe özel karakterler
    for char in punctuation:
        text = text.replace(char, "")

    #Tek karakter ve emoji'leri kaldırma
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  #emoticons
        u"\U0001F300-\U0001F5FF"  #symbols & pictographs
        u"\U0001F680-\U0001F6FF"  #transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  #flags (iOS)
                            "]+", flags=re.UNICODE)

    text = emoji_pattern.sub(r'', text)
    words = re.findall(r'\b\w\b', text)
    for word in words:
        text = text.replace(' ' + word + ' ', ' ')

    return text.strip()

def tokenize(message):
    return [preprocess_text(word) for word in message.split()]

for i, tweet in enumerate(tweets):
    tweets[i] = tokenize(tweet)

#stopwordleri kaldır
from nltk.corpus import stopwords

def remove_stop_words(tweets):
    turkish_stop_words = []
    turkish_stopwords_number = []

    #Türkçe için stop wordler
    stop_words = set(stopwords.words('turkish'))

    for i in tweets:
        sentence = [w for w in i if w.lower() not in stop_words]
        turkish_stopwords_number.append(sentence)
        turkish_stop_words.append(" ".join(sentence))

    return turkish_stop_words, turkish_stopwords_number

tweets, turkish_stopwords_number = remove_stop_words(tweets)

#Uğurhan Şimşir

#öznitelikler csv formatında .txt dosyasına yazılır
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(tweets)
feature_names = vectorizer.get_feature_names_out()

df = pd.DataFrame(tfidf.toarray(), columns=feature_names)

# Tf-idf matrisi bir pandas DataFrame'ine dönüşür ve .txt olarak kaydedilir
df.to_csv('oznitelikler_ugurhan_simsir.txt', index=False)

#Uğurhan Şimşir

from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
import numpy as np

#öznitelik matrisini ve etiketleri yükleme
X = tfidf
y = labels

# y verilerini numpy dizisine dönüştürme
y = np.array(y).astype(int)

#10-fold cross validation nesnesi oluşturma
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

#skf nesnesindeki her bir fold için işlem yapma
scores = []
for train_index, test_index in skf.split(X, y):
    # Eğitim ve test verilerini bölme
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #Multinomial Naive Bayes modeli oluşturma ve eğitme
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    #test setinde doğruluk skoru hesaplar
    accuracy = clf.score(X_test, y_test)
    scores.append(accuracy)

#Uğurhan Şimşir

#test
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Sınıf isimleri
class_names = ['Sınıf 1 (olumlu)', 'Sınıf 2 (olumsuz)', 'Sınıf 3 (nötr)']

#metriklerin tutulacağı listeleri tanımlama
macro_precisions = []
macro_recalls = []
macro_f1_scores = []
micro_precisions = []
micro_recalls = []
micro_f1_scores = []
true_positives = []
false_positives = []
false_negatives = []

#10-fold cross validation ile modeli test etme
for train_index, test_index in skf.split(X, y):

    #eğitim ve test verilerini ayırır
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #Multinomial Naive Bayes modeli oluşturma ve eğitme
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    #test setinde tahmin yapma
    y_pred = clf.predict(X_test)

    #confusion matrix hesaplar
    cm = confusion_matrix(y_test, y_pred)

    #sınıf başına precision, recall, f1-score hesaplar
    precisions = precision_score(y_test, y_pred, average=None)
    recalls = recall_score(y_test, y_pred, average=None)
    f1_scores = f1_score(y_test, y_pred, average=None)

    #sınıf başına true positive, false positive, false negative adedi hesaplar
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    #sınıf başına hesaplanan metrikleri listelere ekler
    macro_precisions.append(precisions.mean())
    macro_recalls.append(recalls.mean())
    macro_f1_scores.append(f1_scores.mean())
    true_positives.append(tp)
    false_positives.append(fp)
    false_negatives.append(fn)

#micro average için confusion matrix hesaplar
cm_micro = np.sum(cm, axis=0)

#micro average için precision, recall, f1-score hesaplar
micro_precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))
micro_recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))
micro_f1_score = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

#macro average için precision, recall, f1-score hesaplar
macro_precision = np.mean(macro_precisions)
macro_recall = np.mean(macro_recalls)
macro_f1_score = np.mean(macro_f1_scores)

#tabloyu ekrana yazdır
print("{:<25} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format("", "Precision", "Recall", "F-Score", "TP", "FP", "FN"))
for i, class_name in enumerate(class_names):
    print("{:<25} {:<15.2f} {:<15.2f} {:<15.2f} {:<15} {:<15} {:<15}".format(class_name,
                                                                              macro_precisions[i],
                                                                              macro_recalls[i],
                                                                              macro_f1_scores[i],
                                                                              np.sum(true_positives[i]),
                                                                              np.sum(false_positives[i]),
                                                                              np.sum(false_negatives[i])))
print("{:<25} {:<15.2f} {:<15.2f} {:<15.2f}".format("Macro Average",
                                                     macro_precision,
                                                     macro_recall,
                                                     macro_f1_score))
print("{:<25} {:<15.2f} {:<15.2f} {:<15.2f}".format("Micro Average",
                                                     micro_precision,
                                                     micro_recall,
                                                     micro_f1_score))

#txt içine yazar
with open('performans_olcum_tablo_ugurhan_simsir.txt', 'w') as file:
    file.write("{:<25} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}\n".format("", "Precision", "Recall", "F-Score", "TP", "FP", "FN"))
    for i, class_name in enumerate(class_names):
        file.write("{:<25} {:<15.2f} {:<15.2f} {:<15.2f} {:<15} {:<15} {:<15}\n".format(class_name,
                                                                                      macro_precisions[i],
                                                                                      macro_recalls[i],
                                                                                      macro_f1_scores[i],
                                                                                      np.sum(true_positives[i]),
                                                                                      np.sum(false_positives[i]),
                                                                                      np.sum(false_negatives[i])))
    file.write("{:<25} {:<15.2f} {:<15.2f} {:<15.2f}\n".format("Macro Average",
                                                               macro_precision,
                                                               macro_recall,
                                                               macro_f1_score))
    file.write("{:<25} {:<15.2f} {:<15.2f} {:<15.2f}\n".format("Micro Average",
                                                               micro_precision,
                                                               micro_recall,
                                                               micro_f1_score))
#Uğurhan Şimşir