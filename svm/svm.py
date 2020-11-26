
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score , roc_auc_score , roc_curve
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score , roc_auc_score , roc_curve
from sklearn.model_selection import train_test_split
import joblib
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score, classification_report
import joblib
import time




def pretreatment():
    root_dir = 'D:\pythonWorkSpace\Bayes\svm'
    d_train = pd.read_csv(root_dir+'\\train.csv')

    d_test = pd.read_csv(root_dir+'\\test.csv')



    label_mapping = {'house':0, 'sports':1, 'yule':2, 'news':3, 'business':4, '2008':5, 'health':6, 'women':7, 'it':8, 'auto':9}
    d_train['category'] = d_train['category'].map(label_mapping)
    d_test['category'] = d_test['category'].map(label_mapping)



    # 将数据集转化为list
    train_articles = d_train['doc'].tolist()
    train_labels = d_train['category'].tolist()
    test_articles = d_test['doc'].tolist()
    test_labels = d_test['category'].tolist()

    # 训练集集转为标准格式
    train_count = len(train_articles)
    for i in range(train_count):
        train_articles[i] = train_articles[i].replace(',', ' ')

    # 测试集转为标准格式
    test_count = len(test_articles)
    for i in range(test_count):
        test_articles[i] = test_articles[i].replace(',', ' ')


    # 存储处理好的数据集
    joblib.dump(train_articles, root_dir+'/Values/train_articles.pkl')
    joblib.dump(test_articles, root_dir+'/Values/test_articles.pkl')
    joblib.dump(train_labels, root_dir+'/Values/train_labels.pkl')
    joblib.dump(test_labels, root_dir+'/Values/test_labels.pkl')

def training():
    root_dir = 'D:\pythonWorkSpace\Bayes\svm'
    train_articles = joblib.load(root_dir+'/Values/train_articles.pkl')
    train_labels = joblib.load(root_dir+'/Values/train_labels.pkl')
    
    

    # 分类器
    text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                    ('clf',LinearSVC())])
    # 训练
    text_clf.fit(train_articles, train_labels)
    joblib.dump(text_clf, root_dir+'./Values/text_clf.pkl')

def predict():
    start = time.time()
    root_dir = 'D:\pythonWorkSpace\Bayes\svm'
    test_articles = joblib.load(root_dir+'/Values/test_articles.pkl')
    test_labels = joblib.load(root_dir+'/Values/test_labels.pkl')

    text_clf = joblib.load(root_dir+'/Values/text_clf.pkl')

    # 预测
    predicted=text_clf.predict(test_articles)


    # 测试集标签
    # print("LinearSVC准确率为：",np.mean(predicted==test_labels))


    # cc = confusion_matrix(test_labels, predicted)
    # pa = precision_score(test_labels, predicted, average=None)
    # ra = recall_score(test_labels, predicted, average=None)
    # f1a = f1_score(test_labels, predicted, average=None)

    print(classification_report(test_labels, predicted))
    end = time.time()
    l_time = end -start
    print ('the time is :%s' %l_time)
    # classes = np.unique(test_labels).tolist()
    # print(classes)



# pretreatment()

# training()

predict()






