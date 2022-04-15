#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt
import nltk
import tensorflow as tf


# In[2]:


df = pd.read_csv('SMSSpamCollection',sep='\t',
                names=['label','message'])


# In[3]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])


# In[4]:


df.duplicated().sum()


# In[5]:


df.drop_duplicates(keep='first')


# In[6]:


df['label'].value_counts()


# In[7]:


plt.pie(df['label'].value_counts(),labels=['ham','spam'],autopct='%0.2f',colors=['r','g'])
plt.show()


# In[8]:


df['no. of sentence'] = df['message'].apply(lambda x:len(nltk.sent_tokenize(x)))
    


# In[9]:


df['no. of words'] = df['message'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[10]:


df['no. of char']  = df['message'].apply(len)


# In[11]:


df.describe()


# In[12]:


df[df['label']==0][['no. of sentence','no. of words','no. of char']].describe()


# In[13]:


df[df['label']==1][['no. of sentence','no. of words','no. of char']].describe()


# In[14]:


import seaborn as sns
plt.figure(figsize=(12,8))
sns.histplot(df[df['label']==0]['no. of words'])
sns.histplot(df[df['label']==1]['no. of words'],color='red')


# In[15]:


plt.figure(figsize=(12,8))
sns.histplot(df[df['label']==0]['no. of sentence'])
sns.histplot(df[df['label']==1]['no. of sentence'],color='red')


# In[16]:


plt.figure(figsize=(12,8))
sns.histplot(df[df['label']==0]['no. of char'])
sns.histplot(df[df['label']==1]['no. of char'],color='red')


# In[17]:


sns.pairplot(df,hue='label')


# In[18]:


df.corr()


# In[19]:


sns.heatmap(df.corr(),annot=True)


# In[20]:


df = df.drop(['no. of sentence','no. of words'],axis=1)


# In[21]:


df


# In[22]:


df = df.drop_duplicates(keep='first')


# In[23]:


df.head(5)


# In[24]:


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
lem = WordNetLemmatizer();
stem = PorterStemmer()
def transform_text(text):
    text.lower()
    words = nltk.word_tokenize(text)
    y = []
    for i in words:
        if (i.isalnum() and i not in stopwords.words('english') and i.isnumeric()!= True):
            x = stem.stem(i)
            y.append(x)
    return ' '.join(y); 


# In[25]:


print(transform_text("Go until clubing pointing craziest jurong point, is the should not in the club crazy.. Available only "))


# In[26]:


df['transformed_message'] = (df['message'].apply(transform_text))


# In[27]:


df


# In[28]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
spam_wc = wc.generate(df[df['label']==1]['transformed_message'].str.cat(sep=" "))


# In[29]:


plt.figure(figsize=(30,15))
plt.imshow(spam_wc)


# In[30]:


y=[]
for i in df[df['label']==1]['transformed_message'].tolist():
    for j in i.split():
        y.append(j)
print(len(y))

z=[]
for i in df[df['label']==0]['transformed_message'].tolist():
    for j in i.split():
        z.append(j)
print(len(z))

from collections import Counter
sns.barplot(pd.DataFrame(Counter(y).most_common(30))[0],pd.DataFrame(Counter(y).most_common(30))[1])
plt.xticks(rotation='vertical')


# In[31]:


sns.barplot(pd.DataFrame(Counter(z).most_common(30))[0],pd.DataFrame(Counter(z).most_common(30))[1])
plt.xticks(rotation='vertical')


# In[32]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_message']).toarray()


# In[33]:


X


# In[34]:


y = df['label'].values


# In[35]:


y


# In[36]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[37]:


from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix
gnb = GaussianNB()
bnb = BernoulliNB()
mnb = MultinomialNB()


# In[38]:


gnb.fit(X_train,y_train)

bnb.fit(X_train,y_train)

mnb.fit(X_train,y_train)


# In[39]:


ypred1 = gnb.predict(X_test)
print(accuracy_score(y_test,ypred1))
print(precision_score(y_test,ypred1))
print(confusion_matrix(y_test,ypred1))


# In[40]:


ypred2 = bnb.predict(X_test)
print(accuracy_score(y_test,ypred2))
print(precision_score(y_test,ypred2))
print(confusion_matrix(y_test,ypred2))


# In[41]:


ypred3 = mnb.predict(X_test)
print(accuracy_score(y_test,ypred3))
print(precision_score(y_test,ypred3))
print(confusion_matrix(y_test,ypred3))


# In[42]:


import pickle
pickle.dump(tfidf,open('vectorized.pkl','wb'))
pickle.dump(transform_text,open('transform.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[43]:


pickle.load(open('vectorized.pkl','rb'))
pickle.load(open('transform.pkl','rb'))
pickle.load(open('model.pkl','rb'))


# In[44]:


pickle.load(open('transform.pkl','rb'))


# In[45]:


pickle.load(open('vectorized.pkl','rb'))


# In[ ]:




