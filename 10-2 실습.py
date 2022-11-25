#!/usr/bin/env python
# coding: utf-8

# In[2]:


type(numbers)


# In[3]:


numbers[-5:]


# In[4]:


numbers.replace(numbers[-5:], '#####')


# In[20]:


def change_num(numbers):
    numbers=numbers.replace(numbers[-5:], '#####')
    return numbers


# In[21]:


change_num('010-13579-24688')


# In[ ]:




