#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[2]:


groceries = []
with open("groceries (2) (1).csv") as f:
    groceries = f.read()
groceries = groceries.split("\n")
groceries


# In[3]:


groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
groceries_list
all_groceries_list = [i for item in groceries_list for i in item]
all_groceries_list


# In[4]:


groceries_series  = pd.DataFrame(pd.Series(groceries_list))
groceries_series = groceries_series.iloc[:9835,:]

groceries_series.columns = ["transactions"]
groceries_series


# In[5]:


X = groceries_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')
X


# In[6]:


frequent_itemsets = apriori(X, min_support=0.005, max_len=3,use_colnames = True)
frequent_itemsets


# In[7]:


frequent_itemsets.sort_values('support',ascending = False,inplace=True)
frequent_itemsets.sort_values


# In[8]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending = False).head(10)


# In[ ]:





# In[29]:


frequent_itemsets_1 = apriori(X, min_support=0.010, max_len=4,use_colnames = True)
frequent_itemsets_1


# In[30]:


frequent_itemsets_1.sort_values('support',ascending = False,inplace=True)
frequent_itemsets_1.sort_values


# In[31]:


rules_1 = association_rules(frequent_itemsets_1, metric="lift", min_threshold=1)
rules_1.head(20)
rules_1.sort_values('lift',ascending = False).head(10)


# In[ ]:




