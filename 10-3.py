#!/usr/bin/env python
# coding: utf-8

# In[7]:


a = [[1, 2], 3, [[4, 5, 6], 7], 8, 9]
def flatten(data):
    return output


# In[8]:


type(a[0])


# In[13]:


def flatten(data):
    output = []  # 빈 리스트를 만듭니다.
    for item in data:
        if type(item) == list:
            output += flatten(item)
        else:
            output.append(item)
    return output


# In[14]:


a = [[1, 2], 3, [[4, 5, 6], 7], 8, 9]
flatten(a)


# In[ ]:




