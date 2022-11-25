#!/usr/bin/env python
# coding: utf-8

# In[1]:


x = 10

if type(x) == int:
    print('정수입니다')


# In[2]:


x = 10.0

if type(x) == int:
    print('정수입니다')


# In[3]:


x = 10

if x % 1 == 0:
    print('정수입니다')


# In[4]:


def int_divider(x, y):
    if x %1 == 0 and y%1 ==0:
        answer=x/y
    else:
        print('정수만 입력하세요.')
        answer=None
    return answer


# In[5]:


int_divider(10,3)


# In[7]:


int_divider(-1, 0.9)


# In[8]:


int_divider(6.0, 2)


# In[9]:


def mul(*values):
    return output


# In[10]:


def mul(*values):
    return output


# In[11]:


def mul(*values):
    output = 1
    for num in values:
        if num <= 10:
            output *= num


# In[12]:


def mul(*values):
    output = 1
    for num in values:
        if num <= 10:
            output *= num
        else:
            pass
    return output


# In[13]:


mul(2, 3, 4, 5)


# In[14]:


mul(3, 12, 10)


# In[15]:


mul(3, 12, 10)


# In[ ]:




