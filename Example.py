#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import datetime
import requests_cache
from datetime import timedelta
import datetime


# In[2]:


# #загружаю данные с финама
# s = pd.read_csv('AFLT_170605_210605.csv')
# stock = s.iloc[:, 2:]
# stock.rename({'<DATE>': 'date','<OPEN>': 'open', '<HIGH>': 'high', '<LOW>': 'low', '<CLOSE>': 'close',
#           '<VOL>': 'vol'}, axis=1, inplace=True)
# l=[]
# #меняю даты на нужный формат
# for i in range(len(stock.date)):
#     p=str(stock.date[i])
#     oldformat = p
#     datetimeobject = datetime.strptime(oldformat,'%Y%m%d')
#     q=datetimeobject.strftime('%Y-%m-%d')
#     l.append(q)
# ind=pd.DatetimeIndex(l)
# stock.index = ind
# df=stock
# df=df.drop(df.columns[[0, 1]], axis=1)
# df #все данные с акции

stock_name = 'TSLA'

start = datetime.datetime(2017, 6, 1)
end = datetime.datetime(2020, 12, 1)

try:
    expire_after = datetime.timedelta(days=3)
    session = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after=expire_after)
    data_exchange =  web.DataReader(stock_name, 'yahoo', start, end, session=session)  
except: 
    print('Этих акции на такие даты нет')
    
data_exchange = data_exchange.drop(columns = 'Adj Close')
data_exchange.columns = ['open','high','low','close','vol']
df = data_exchange
df


# In[3]:


def pivot_tikers(df,col):
    item=[]
    tikers=[]
    data=df[col]
    item.append(data)
    combine=pd.concat(item, axis=1) 
    combine.columns=[stock_name]
    return combine  


# In[4]:


#беру открытие
df_open = pivot_tikers(df,'open')
ma_200 = df_open.rolling(45).mean() #пробовала 150 и 200 на двух файлах
ma_50 = df_open.rolling(15).mean()
result=pd.concat([df_open,ma_200,ma_50],axis=1)
result.columns=[stock_name,'ma_200','ma_50']
result #таблица со скользящими и ценами акции по датам


# In[5]:


#заполняю NaN разными числами, чтобы при сравнении строчки не совпадали
rrr = result.copy()
rrr['ma_200'] = rrr['ma_200'].fillna(7)
rrr['ma_50'] = rrr['ma_50'].fillna(9)
rrr 


# In[6]:


#ищу пересечение
idx = np.argwhere(np.diff(np.sign(rrr["ma_200"] - rrr["ma_50"]))).flatten()
if rrr.iloc[idx[0]][1]==7 or rrr.iloc[idx[0]][2]==9:
    lk = idx[1:]
else:
    lk = idx
lk1 = lk+1 #чтобы покупать и продавать акции на следующий день после пересечения
#разделяем даты на покупку и продажу
lsa = []
for i in range(0, len(lk), 2):
    o = lk[i]
    lsa.append(o)
lsd= []
for i in range(0, len(lk)):
    if i % 2 != 0:
        z =lk[i]
        lsd.append(z)

        
        
plt.figure(figsize=(15, 5), dpi=80)
plt.plot(result.index[lsa], result.loc[result.index[lsa]]['ma_200'],'v', markersize=10, color='k')
plt.plot(result.index[lsd], result.loc[result.index[lsd]]['ma_200'],'v', markersize=10, color='m')
plt.plot(result.index, result.values)
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.title('Динамика цены акции')
plt.legend(["buy", "sell", stock_name, "ma_200", "ma_50"])
plt.show()


# In[7]:


df_next = pd.DataFrame(df_open[stock_name][result.index[lk1]]) #датафрейм
nextt = df_open[stock_name][result.index[lk1]] #Series
nextt


# In[8]:


#работа со стоп-лоссом
stoploss = 0.1
seriestake = df_open[stock_name][result.index[:]] 
seriq = pd.Series(nextt.iloc[[0]])
if len(nextt) % 2 ==0:
    for j in range(len(lk1)):
        if j % 2 ==0:
            for i in range(len(seriestake)-1):
                if (i > lk1[j] and df_open[stock_name][result.index[i]] < df_open[stock_name][result.index[lk1[j]]]*(1-stoploss) and i<lk1[j+1] and seriestake.iloc[[i]][0]>nextt[j+1]):
                    seriq = seriq.append(nextt.iloc[[j]])
                    seriq = seriq.append(seriestake.iloc[[i]])
                    break
                elif i == lk1[j+1]:
                    seriq = seriq.append(nextt.iloc[[j]])
                    seriq = seriq.append(seriestake.iloc[[lk1[j+1]]])
else:
    for j in range(len(lk1)):
        if (j % 2 ==0 and j != len(lk1)-1):
            for i in range(len(seriestake)-1):
                if (i > lk1[j] and df_open[stock_name][result.index[i]] < df_open[stock_name][result.index[lk1[j]]]*(1-stoploss) and i<lk1[j+1] and seriestake.iloc[[i]][0]>nextt[j+1]):
                    seriq = seriq.append(nextt.iloc[[j]])
                    seriq = seriq.append(seriestake.iloc[[i]])
                    break
                elif i == lk1[j+1]:
                    seriq = seriq.append(nextt.iloc[[j]])
                    seriq = seriq.append(seriestake.iloc[[lk1[j+1]]])
        elif j == len(lk1)-1:
            for i in range(len(seriestake)-1):
                if (i > lk1[j] and df_open[stock_name][result.index[i]] < df_open[stock_name][result.index[lk1[j]]]*(1-stoploss)):
                    seriq = seriq.append(nextt.iloc[[j]])
                    seriq = seriq.append(seriestake.iloc[[i]])
                    break  
                elif i == (len(seriestake)-2):
                    seriq = seriq.append(nextt.iloc[[j]])
                    seriq = seriq.append(seriestake.iloc[[len(seriestake)-2]])
seriq      
endd = seriq.iloc[1:]
df_end = pd.DataFrame(endd)
endd #получили новую талицу для дней покупки и продажи - учитывая стоп-лоссы


# In[9]:


#считаем доходность с покупки и последующей продажи каждый раз
lissq = []
for i in range(len(endd)-1):
    lllq=(endd[i+1]-endd[i])/ endd[i]*100
    lissq.append(lllq)
for i in range(len(lissq)):
    if i % 2 !=0:
        lissq[i] = lissq[i]*0
lissq
lissq.insert(0,0)
lissq #массив с доходностью


# In[10]:


#итоговая таблица 
df_end['Buy or Sell'] = ['Buy', 'Sell']*int(len(endd)/2)
df_end['Profit_pc'] = lissq
df_end


# In[11]:


#Вычисляем прибыль в денежных едеиницах, если каждую акцию покупаем на все деньги
profit_itog=0
for i in range(len(df_end)):
    if i % 2==0:
        get = df_end[stock_name][i+1] - df_end[stock_name][i] 
        profit_itog+=get
profit_itog


# In[ ]:




