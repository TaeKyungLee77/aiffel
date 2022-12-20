#!/usr/bin/env python
# coding: utf-8

# In[ ]:


라이브러리 버전을 확인


# In[24]:


import sklearn

print(sklearn.__version__)


# In[ ]:


#(1) 필요한 모듈 import하기#


# In[25]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[ ]:


#(2) 데이터 준비#


# In[32]:


from sklearn.datasets import load_digits

digits = load_digits()
print(dir(digits))


# In[ ]:


#(3) 데이터 이해하기#


# In[28]:


digits.keys()


# In[30]:


digits_data = digits.data

print(digits_data.shape)


# In[31]:


digits_data[0]


# In[34]:


digits_label = digits.target

print(digits_label.shape)
digits_label


# In[35]:


digits.target_names


# In[36]:


print(digits.DESCR)


# In[37]:


digits.feature_names


# In[41]:


type(digits_data)


# In[42]:


digits_df = pd.DataFrame(data=digits_data, columns=digits.feature_names)
digits_df


# In[43]:


digits_df["label"] = digits.target

digits_df


# In[44]:


#(4) train, test 데이터 분리#


# In[45]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(digits_data, 
                                                    digits_label, 
                                                    test_size=0.2, 
                                                    random_state=7)

print('X_train 개수: ', len(X_train),', X_test 개수: ', len(X_test))


# In[46]:


X_train.shape, y_train.shape


# In[47]:


X_test.shape, y_test.shape


# In[48]:


y_train, y_test


# In[51]:


#(5) 다양한 모델로 학습시켜보기#


# In[ ]:


의사결정 나무


# In[52]:


from sklearn.tree import DecisionTreeClassifier 

decision_tree = DecisionTreeClassifier(random_state=32) 
print(decision_tree._estimator_type)


# In[53]:


decision_tree.fit(X_train, y_train)


# In[54]:


y_pred = decision_tree.predict(X_test)
y_pred


# In[55]:


y_test


# In[56]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[ ]:


랜덤 포레스트


# In[57]:


from sklearn.ensemble import RandomForestClassifier 

X_train, X_test, y_train, y_test = train_test_split(digits_data, 
                                                    digits_label, 
                                                    test_size=0.2, 
                                                    random_state=21) 

random_forest = RandomForestClassifier(random_state=32) 
random_forest.fit(X_train, y_train) 
y_pred = random_forest.predict(X_test) 

print(classification_report(y_test, y_pred)) 


# In[58]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[ ]:


SVM


# In[59]:


from sklearn import svm 
svm_model = svm.SVC() 

print(svm_model._estimator_type) 


# In[60]:


svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test) 

print(classification_report(y_test, y_pred)) 


# In[61]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[ ]:


SGD


# In[62]:


from sklearn.linear_model import SGDClassifier 
sgd_model = SGDClassifier() 

print(sgd_model._estimator_type) 


# In[63]:


sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test) 

print(classification_report(y_test, y_pred)) 


# In[64]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[65]:


from sklearn.linear_model import LogisticRegression 
logistic_model = LogisticRegression() 

print(logistic_model._estimator_type)


# In[66]:


logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test) 

print(classification_report(y_test, y_pred))


# In[67]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[ ]:


SVM 이 가장 높은 정확도를 보여줌

