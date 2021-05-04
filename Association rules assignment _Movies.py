#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules


# In[13]:


movies= pd.read_csv("my_movies.csv")
movies.head()
movies_new= movies.drop(columns=['V1', 'V2', 'V3', 'V4', 'V5'])
movies_new.head()


# In[14]:


frequent_itemsets = apriori(movies_new, min_support=0.002, max_len=3,use_colnames = True)
frequent_itemsets


# In[15]:


frequent_itemsets.sort_values('support',ascending = False,inplace=True)
frequent_itemsets.sort_values


# In[16]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending = False).head(10)


# In[17]:


frequent_itemsets_1 = apriori(movies_new, min_support=0.004, max_len=4,use_colnames = True)
frequent_itemsets_1


# In[18]:


frequent_itemsets_1.sort_values('support',ascending = False,inplace=True)
frequent_itemsets_1.sort_values


# In[19]:


rules = association_rules(frequent_itemsets_1, metric="lift", min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending = False).head(10)


# In[ ]:




