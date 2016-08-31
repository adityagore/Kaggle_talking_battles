#==============================================================================
# Don't forget to change the working directory before running
#==============================================================================

#==============================================================================
# Import Packages
# 
#==============================================================================
import pandas as pd
import numpy as np
#matplotlib inline
#import seaborn as sns
#import matplotlib.pyplot as plt
import os
from scipy.sparse import csr_matrix, hstack
from datetime import date
mingw_path = 'C:\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss

import xgboost as xgb

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, ActivityRegularization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2, activity_l1l2


import re
#==============================================================================
# Define Functions
#==============================================================================

def rstr(df): return df.dtypes, df.head(3) ,df.apply(lambda x: [x.unique()]), df.apply(lambda x: [len(x.unique())]),df.shape

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

seed = 1851
np.random.seed(seed)  

#==============================================================================
# Load Data
#==============================================================================

datadir = ''
gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),
                      index_col='device_id')
gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'),
                     index_col = 'device_id')
            
gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])        
gatrain.head()

phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))
phone.head()

# In[ ]:

phone.shape


# In[ ]:

phone.duplicated().sum()


# In[ ]:

phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')
phone.shape


# In[ ]:

phone.head()


# In[ ]:

events = pd.read_csv(os.path.join(datadir,'events.csv'), parse_dates=['timestamp'])
events.head()


# In[ ]:

appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'),usecols=['event_id','app_id','is_active'],dtype={'is_active':bool})
appevents.head()


# In[ ]:

applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))
applabels.head()


# In[ ]:

labelcategories = pd.read_csv(os.path.join(datadir,'label_categories.csv'))
labelcategories.head()

app_labels = applabels.merge(labelcategories,how='left',left_on='label_id',right_on='label_id')
app_labels['general_groups'] = app_labels['category']

def to_Games(x):
    # I have gone through 'almost' manually through existing categories and came up with this non-elegant regex
    if re.search('([gG]am)|([pP]oker)|([cC]hess)|([pP]uzz)|([bB]all)|([pP]ursu)|([fF]ight)|([sS]imulat)|([sS]hoot)', 
                 x) is not None:
        return('Games')
    # Then I went through existing abbreviations like RPG, MMO and so on
    if re.search('(RPG)|(SLG)|(RAC)|(MMO)|(MOBA)', x) is not None:
        return('Games')
    # Still small list of items left which is not covered by regex
    if x in ['billards', 'World of Warcraft', 'Tower Defense', 'Tomb', 'Ninja', 'Europe and Fantasy', 'Senki', 
             'Shushan', 'Lottery ticket', 'majiang', 'tennis', 'Martial arts']:
        return('Games')
    else:
        return(x)
        


app_labels['general_groups'] = app_labels['general_groups'].apply(to_Games)


        
def to_Property(x):
    # All property/estate stuff will be place into Property category
    if x in ['Property Industry 2.0', 'Property Industry new', 'Property Industry 1.0']:
        return('Property')
    if re.search('([eE]state)', x) is not None:
        return('Property')
    else:
        return(x)
        
app_labels['general_groups'] = app_labels['general_groups'].apply(to_Property)

def to_Family(x):
    if re.search('([fF]amili)|([mM]othe)|([fF]athe)|(bab)|([rR]elative)|([pP]regnan)|([pP]arent)|([mM]arriag)|([lL]ove)',
                 x) is not None:
        return('Family')
    else:
        return(x)

app_labels['general_groups'] = app_labels['general_groups'].apply(to_Family)

def to_Fun(x):
    '''One can argue about my decision however I used following rules:
       - all comics -> Fun
       - all animation/painting -> Fun
       - all things labeled as trend or passion or community -> Fun
       - all things I could identify as messangers -> Fun
       - all things related to images/pictures -> Fun
       - horoscopes -> Fun
       - jokes -> Fun
       - I don\'t know what is Parkour avoid but it goes to -> Fun''' 
    if re.search('([fF]un)|([cC]ool)|([tT]rend)|([cC]omic)|([aA]nima)|([pP]ainti)|\
                 ([fF]iction)|([pP]icture)|(joke)|([hH]oroscope)|([pP]assion)|([sS]tyle)|\
                 ([cC]ozy)|([bB]log)', x) is not None:
        return('Fun')
    if x in ['Parkour avoid class', 'community', 'Enthusiasm', 'cosplay', 'IM']:
        return('Fun')
    else:
        return(x)

app_labels['general_groups'] = app_labels['general_groups'].apply(to_Fun)

def to_Productivity(x):
    if x == 'Personal Effectiveness 1' or x == 'Personal Effectiveness':
        return('Productivity')
    else:
        return(x)

app_labels['general_groups'] = app_labels['general_groups'].apply(to_Productivity)

def to_Finance(x):
    if re.search('([iI]ncome)|([pP]rofitabil)|([lL]iquid)|([rR]isk)|([bB]ank)|([fF]uture)|([fF]und)|([sS]tock)|([sS]hare)',
                 x) is not None:
        return('Finance')
    if re.search('([fF]inanc)|([pP]ay)|(P2P)|([iI]nsura)|([lL]oan)|([cC]ard)|([mM]etal)|\
                  ([cC]ost)|([wW]ealth)|([bB]roker)|([bB]usiness)|([eE]xchange)', x) is not None:
        return('Finance')
    if x in ['High Flow', 'Housekeeping', 'Accounting', 'Debit and credit', 'Recipes', 'Heritage Foundation', 'IMF',]:
        return('Finance')
    else:
        return(x)

app_labels['general_groups'] = app_labels['general_groups'].apply(to_Finance)

def to_Religion(x):
    if x == 'And the Church':
        return('Religion')
    else:
        return(x)
        
app_labels['general_groups'] = app_labels['general_groups'].apply(to_Religion)

def to_Services(x):
    if re.search('([sS]ervice)', x) is not None:
        return('Services')
    else:
        return(x)

app_labels['general_groups'] = app_labels['general_groups'].apply(to_Services)

def to_Travel(x):
    if re.search('([aA]viation)|([aA]irlin)|([bB]ooki)|([tT]ravel)|\
                  ([hH]otel)|([tT]rain)|([tT]axi)|([rR]eservati)|([aA]ir)|([aA]irport)', x) is not None:
        return('Travel')
    if re.search('([jJ]ourne)|([tT]ransport)|([aA]ccommodat)|([nN]avigat)|([tT]ouris)|([fF]light)|([bB]us)', x) is not None:
        return('Travel')
    if x in ['High mobility', 'Destination Region', 'map', 'Weather', 'Rentals']:
        return('Travel')
    else:
        return(x)

app_labels['general_groups'] = app_labels['general_groups'].apply(to_Travel)

def to_Custom(x):
    if re.search('([cC]ustom)', x) is not None:
        return('Custom')
    else:
        return(x)

app_labels['general_groups'] = app_labels['general_groups'].apply(to_Custom)

def to_Video(x):
    # not sure if round means Rounds app for group chat, but I stick to this hypothesis. Might be popular app in China
    if x in ['video', 'round', 'the film', 'movie']:
        return('Video')
    else:
        return(x)
        
app_labels['general_groups'] = app_labels['general_groups'].apply(to_Video)
        


def to_Shopping(x):
    if x in ['Smart Shopping', 'online malls', 'online shopping by group, like groupon', 'takeaway ordering', 
             'online shopping, price comparing', 'Buy class', 'Buy', 'shopping sharing',
             'Smart Shopping 1', 'online shopping navigation']:
        return('Shopping')
    else:
        return(x)

app_labels['general_groups'] = app_labels['general_groups'].apply(to_Shopping)

def to_Education(x):
    if re.search('([eE]ducati)|([rR]ead)|([sS]cienc)|([bB]ooks)', x) is not None:
        return('Education')
    if x in ['literature', 'Maternal and child population', 'psychology', 'exams', 'millitary and wars', 'news', 
             'foreign language', 'magazine and journal', 'dictionary', 'novels', 'art and culture', 'Entertainment News', 
             'College Students', 'math', 'Western Mythology', 'Technology Information', 'study abroad', 
             'Chinese Classical Mythology']:
        return('Education')
    else:
        return(x)

app_labels['general_groups'] = app_labels['general_groups'].apply(to_Education)

def to_Vitality(x):
    if x in ['vitality', '1 vitality']:
        return('Vitality')
    if x in [ 'sports and gym', 'Health Management', 'Integrated Living', 'Medical', 'Free exercise', 'A beauty care', 
             'fashion', 'fashion outfit', 'lose weight', 'health', 'Skin care applications', 'Wearable Health']:
        return('Vitality')
    else:
        return(x)
 
app_labels['general_groups'] = app_labels['general_groups'].apply(to_Vitality)
       
def to_Sports(x):
    if x in ['sports', 'Sports News']:
        return('Sports')
    else:
        return(x)

app_labels['general_groups'] = app_labels['general_groups'].apply(to_Sports)
    
def to_Music(x):
    if x == 'music':
        return('Music')
    else:
        return(x)

app_labels['general_groups'] = app_labels['general_groups'].apply(to_Music)

def to_Travel_2(x):
    if re.search('([hH]otel)', x) is not None:
        return('Travel')
    else:
        return(x)

app_labels['general_groups'] = app_labels['general_groups'].apply(to_Travel_2)

def to_Other(x):
    if x in ['1 free',
 'The elimination of class',
 'unknown',
 'free',
 'comfortable',
 'Cozy 1',
 'other',
 'Total Cost 1',
 'Classical 1',
 'Quality 1',
 'classical',
 'quality',
 'Car Owners',
 'Noble 1',
 'Pirated content',
 'Securities',
 'professional skills',
 'Jobs',
 'Reputation',
 'Simple 1',
 '1 reputation',
 'Condition of the vehicles',
 'magic',
 'Internet Securities',
 'weibo',
 'Housing Advice',
 'notes',
 'farm',
 'Nature 1',
 'Total Cost',
 'Sea Amoy',
 'show',
 'Car',
 'pet raising up',
 'dotal-lol',
 'Express',
 'radio',
 'Occupational identity',
 'Utilities',
 'Trust',
 'Contacts',
 'Simple',
 'Automotive News',
 'Sale of cars',
 'File Editor',
 'network disk',
 'class managemetn',
 'management',
 'natural',
 'Points Activities',
 'Decoration',
 'store management',
 'Maternal and child supplies',
 'Tour around',
 'coupon',
 'User Community',
 'Vermicelli',
 'noble',
 'poetry',
 'Antique collection',
 'Reviews',
 'Scheduling',
 'Beauty Nail',
 'shows',
 'Hardware Related',
 'Smart Home',
 'Sellers',
 'Desktop Enhancements',
 'library',
 'entertainment',
 'Calendar',
 'Ping',
 'System Tools',
 'KTV',
 'Behalf of the drive',
 'household products',
 'Information',
 'Man playing favorites',
 'App Store',
 'Engineering Drawing',
 'Academic Information',
 'Appliances',
 'Peace - Search',
 'Make-up application',
 'WIFI',
 'phone',
 'Doctors',
 'Smart Appliances',
 'reality show',
 'Harem',
 'trickery',
 'Jin Yong',
 'effort',
 'Xian Xia',
 'Romance',
 'tribe',
 'email',
 'mesasge',
 'Editor',
 'Clock',
 'search',
 'Intelligent hardware',
 'Browser',
 'Furniture']:
        return('Other')
    else:
        return(x)

app_labels['general_groups'] = app_labels['general_groups'].apply(to_Other)

# In[ ]:

print('Number of unique categores: {}, Number of unique categores in apps {}, Number of categories that match {}'.format(labelcategories.shape[0],len(app_labels.label_id.unique()),labelcategories.label_id.isin(app_labels.label_id).sum()))


# # Device Brand

# In[ ]:

brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gatrain['brand'] = phone['brand']
gatest['brand'] = phone['brand']
Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]), (gatrain.trainrow, gatrain.brand)))
Xte_brand = csr_matrix((np.ones(gatest.shape[0]), (gatest.testrow, gatest.brand)))
print('Brand features: train shape {}, test shape {}'.format(Xtr_brand.shape, Xte_brand.shape))


# # Device Model

# In[ ]:

m = phone.phone_brand.str.cat(phone.device_model)
m.head()


# In[ ]:

modelencoder = LabelEncoder().fit(m)
phone['model'] = modelencoder.transform(m)
gatrain['model'] = phone['model']
gatest['model'] = phone['model']
Xtr_model = csr_matrix((np.ones(gatrain.shape[0]), (gatrain.trainrow, gatrain.model)))
Xte_model = csr_matrix((np.ones(gatest.shape[0]), (gatest.testrow, gatest.model)))
print('Model features: train shape {}, test shape {}'.format(Xtr_model.shape, Xte_model.shape))


# In[ ]:

### Apps

appencoder = LabelEncoder().fit(appevents.app_id)
appevents['app'] = appencoder.transform(appevents.app_id)
napps = len(appencoder.classes_)
appevents.head()


# In[ ]:

events.head()


# In[ ]:

latLong = pd.read_csv("latLong.csv")
latLong.head()


# In[ ]:

events = events.merge(latLong[['longitude','latitude','regionLabel']],how='left',left_on=['longitude','latitude'],right_on=['longitude','latitude'])
events.head()


# In[ ]:

events = events.drop(['longitude','latitude'],axis=1)
events.head()


# In[ ]:

events.dtypes


# In[ ]:

events['day_of_week']=events.timestamp.dt.dayofweek
events['time_of_day']=events.timestamp.dt.hour
events['period']=pd.cut(events.time_of_day,bins=[0,5,12,17,21,24],right=False,labels=[0,1,2,3,4])
events['period'].replace(4,0,inplace=True)
# 0:Night,1:Morning,2:Afternoon,3:Evening
events =events.drop(['time_of_day','timestamp'],axis=1)
events.head()


# In[ ]:

appevents.head()


# In[ ]:

appencoder = LabelEncoder().fit(appevents.app_id)
appevents['app'] = appencoder.transform(appevents.app_id)
napps = len(appencoder.classes_)
appevents.head()


deviceapps=appevents.merge(events,how='left',left_on=['event_id'],right_on=['event_id'])
deviceapps.head()


# # Apps and Activity

# In[ ]:

appactivity = deviceapps.groupby(['device_id','app'])['is_active'].agg(['size',np.sum,np.mean]).reset_index()
appactivity=(appactivity.merge(gatrain[['trainrow']],how='left',left_on='device_id',right_index=True)
.merge(gatest[['testrow']],how='left',left_on='device_id',right_index=True))
appactivity.head()


# In[ ]:

d = appactivity.dropna(subset=['trainrow'])
Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)), 
                      shape=(gatrain.shape[0],napps))
d = appactivity.dropna(subset=['testrow'])
Xte_app = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.app)), 
                      shape=(gatest.shape[0],napps))
print('Apps data: train shape {}, test shape {}'.format(Xtr_app.shape, Xte_app.shape))


# In[ ]:

d = appactivity.dropna(subset=['trainrow'])
Xtr_app_size = csr_matrix((d['size'], (d.trainrow, d.app)), 
                      shape=(gatrain.shape[0],napps))
d = appactivity.dropna(subset=['testrow'])
Xte_app_size = csr_matrix((d['size'], (d.testrow, d.app)), 
                      shape=(gatest.shape[0],napps))
print('Apps data: train shape {}, test shape {}'.format(Xtr_app_size.shape, Xte_app_size.shape))


# In[ ]:

d = appactivity.dropna(subset=['trainrow'])
Xtr_app_act = csr_matrix((np.log1p(d['sum']), (d.trainrow, d.app)), 
                      shape=(gatrain.shape[0],napps))
d = appactivity.dropna(subset=['testrow'])
Xte_app_act = csr_matrix((np.log1p(d['sum']), (d.testrow, d.app)), 
                      shape=(gatest.shape[0],napps))
print('Apps data: train shape {}, test shape {}'.format(Xtr_app_act.shape, Xte_app_act.shape))


# In[ ]:

d = appactivity.dropna(subset=['trainrow'])
Xtr_app_act_m = csr_matrix((d['mean'], (d.trainrow, d.app)), 
                      shape=(gatrain.shape[0],napps))
d = appactivity.dropna(subset=['testrow'])
Xte_app_act_m = csr_matrix((d['mean'], (d.testrow, d.app)), 
                      shape=(gatest.shape[0],napps))
print('Apps data: train shape {}, test shape {}'.format(Xtr_app_act_m.shape, Xte_app_act_m.shape))


# # Location and Activity

# In[ ]:

locactivity = deviceapps.groupby(['device_id','regionLabel'])['is_active'].agg(['size',np.sum,np.mean]).reset_index()
locactivity=(locactivity.merge(gatrain[['trainrow']],how='left',left_on='device_id',right_index=True)
.merge(gatest[['testrow']],how='left',left_on='device_id',right_index=True))
locactivity.head()


# In[ ]:

locEncoder = LabelEncoder().fit(locactivity['regionLabel'])
locactivity['loc'] = locEncoder.transform(locactivity['regionLabel'])
nlocs = len(locEncoder.classes_)
locactivity.head()


# In[ ]:

d = locactivity.dropna(subset=['trainrow'])
Xtr_loc = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d['loc'])), 
                      shape=(gatrain.shape[0],nlocs))
d = locactivity.dropna(subset=['testrow'])
Xte_loc = csr_matrix((np.ones(d.shape[0]), (d.testrow, d['loc'])), 
                      shape=(gatest.shape[0],nlocs))
print('Location data: train shape {}, test shape {}'.format(Xtr_loc.shape, Xte_loc.shape))


# In[ ]:

d = locactivity.dropna(subset=['trainrow'])
Xtr_loc_size = csr_matrix((d['size'], (d.trainrow, d['loc'])), 
                      shape=(gatrain.shape[0],nlocs))
d = locactivity.dropna(subset=['testrow'])
Xte_loc_size = csr_matrix((d['size'], (d.testrow, d['loc'])), 
                      shape=(gatest.shape[0],nlocs))
print('Location data: train shape {}, test shape {}'.format(Xtr_loc_size.shape, Xte_loc_size.shape))


# In[ ]:

d = locactivity.dropna(subset=['trainrow'])
Xtr_loc_act = csr_matrix((np.log1p(d['sum']), (d.trainrow, d['loc'])), 
                      shape=(gatrain.shape[0],nlocs))
d = locactivity.dropna(subset=['testrow'])
Xte_loc_act = csr_matrix((np.log1p(d['sum']), (d.testrow, d['loc'])), 
                      shape=(gatest.shape[0],nlocs))
print('Location data: train shape {}, test shape {}'.format(Xtr_loc_act.shape, Xte_loc_act.shape))


# In[ ]:

d = locactivity.dropna(subset=['trainrow'])
Xtr_loc_act_m = csr_matrix((d['mean'], (d.trainrow, d['loc'])), 
                      shape=(gatrain.shape[0],nlocs))
d = locactivity.dropna(subset=['testrow'])
Xte_loc_act_m = csr_matrix((d['mean'], (d.testrow, d['loc'])), 
                      shape=(gatest.shape[0],nlocs))
print('Location data: train shape {}, test shape {}'.format(Xtr_loc_act_m.shape, Xte_loc_act_m.shape))


# # Week day and Activity

# In[ ]:

weekactivity = deviceapps.groupby(['device_id','day_of_week'])['is_active'].agg(['size',np.sum,np.mean]).reset_index()
weekactivity=(weekactivity.merge(gatrain[['trainrow']],how='left',left_on='device_id',right_index=True)
.merge(gatest[['testrow']],how='left',left_on='device_id',right_index=True))
weekactivity.head()


# In[ ]:

d = weekactivity.dropna(subset=['trainrow'])
Xtr_week = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d['day_of_week'])), 
                      shape=(gatrain.shape[0],7))
d = weekactivity.dropna(subset=['testrow'])
Xte_week = csr_matrix((np.ones(d.shape[0]), (d.testrow, d['day_of_week'])), 
                      shape=(gatest.shape[0],7))
print('Week data: train shape {}, test shape {}'.format(Xtr_week.shape, Xte_week.shape))


# In[ ]:

d = weekactivity.dropna(subset=['trainrow'])
Xtr_week_size = csr_matrix((d['size'], (d.trainrow, d['day_of_week'])), 
                      shape=(gatrain.shape[0],7))
d = weekactivity.dropna(subset=['testrow'])
Xte_week_size = csr_matrix((d['size'], (d.testrow, d['day_of_week'])), 
                      shape=(gatest.shape[0],7))
print('Week data: train shape {}, test shape {}'.format(Xtr_week_size.shape, Xte_week_size.shape))


# In[ ]:

d = weekactivity.dropna(subset=['trainrow'])
Xtr_week_act = csr_matrix((np.log1p(d['sum']), (d.trainrow, d['day_of_week'])), 
                      shape=(gatrain.shape[0],7))
d = weekactivity.dropna(subset=['testrow'])
Xte_week_act = csr_matrix((np.log1p(d['sum']), (d.testrow, d['day_of_week'])), 
                      shape=(gatest.shape[0],7))
print('Week data: train shape {}, test shape {}'.format(Xtr_week_act.shape, Xte_week_act.shape))


# In[ ]:

d = weekactivity.dropna(subset=['trainrow'])
Xtr_week_act_m = csr_matrix((d['mean'], (d.trainrow, d['day_of_week'])), 
                      shape=(gatrain.shape[0],7))
d = weekactivity.dropna(subset=['testrow'])
Xte_week_act_m = csr_matrix((d['mean'], (d.testrow, d['day_of_week'])), 
                      shape=(gatest.shape[0],7))
print('Week data: train shape {}, test shape {}'.format(Xtr_week_act_m.shape, Xte_week_act_m.shape))


# # Day time & Activity

# In[ ]:

periodactivity = deviceapps.groupby(['device_id','period'])['is_active'].agg(['size',np.sum,np.mean]).reset_index()
periodactivity=(periodactivity.merge(gatrain[['trainrow']],how='left',left_on='device_id',right_index=True)
.merge(gatest[['testrow']],how='left',left_on='device_id',right_index=True))
periodactivity.head()


# In[ ]:

d = periodactivity.dropna(subset=['trainrow'])
Xtr_period = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d['period'])), 
                      shape=(gatrain.shape[0],4))
d = periodactivity.dropna(subset=['testrow'])
Xte_period = csr_matrix((np.ones(d.shape[0]), (d.testrow, d['period'])), 
                      shape=(gatest.shape[0],4))
print('Period data: train shape {}, test shape {}'.format(Xtr_period.shape, Xte_period.shape))


# In[ ]:

d = periodactivity.dropna(subset=['trainrow'])
Xtr_period_size = csr_matrix((d['size'], (d.trainrow, d['period'])), 
                      shape=(gatrain.shape[0],4))
d = periodactivity.dropna(subset=['testrow'])
Xte_period_size = csr_matrix((d['size'], (d.testrow, d['period'])), 
                      shape=(gatest.shape[0],4))
print('Period data: train shape {}, test shape {}'.format(Xtr_period_size.shape, Xte_period_size.shape))


# In[ ]:

d = periodactivity.dropna(subset=['trainrow'])
Xtr_period_act = csr_matrix((np.log1p(d['sum']), (d.trainrow, d['period'])), 
                      shape=(gatrain.shape[0],4))
d = periodactivity.dropna(subset=['testrow'])
Xte_period_act = csr_matrix((np.log1p(d['sum']), (d.testrow, d['period'])), 
                      shape=(gatest.shape[0],4))
print('Period data: train shape {}, test shape {}'.format(Xtr_period_act.shape, Xte_period_act.shape))


# In[ ]:

d = periodactivity.dropna(subset=['trainrow'])
Xtr_period_act_m = csr_matrix((d['mean'], (d.trainrow, d['period'])), 
                      shape=(gatrain.shape[0],4))
d = periodactivity.dropna(subset=['testrow'])
Xte_period_act_m = csr_matrix((d['mean'], (d.testrow, d['period'])), 
                      shape=(gatest.shape[0],4))
print('Period data: train shape {}, test shape {}'.format(Xtr_period_act_m.shape, Xte_period_act_m.shape))


# In[ ]:

del appactivity
del locactivity
del weekactivity
del periodactivity
del events


# # App labels features

# In[ ]:

#==============================================================================
# Old Version
#==============================================================================
# applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]
# applabels['app'] = appencoder.transform(applabels.app_id)
# labelencoder = LabelEncoder().fit(applabels.label_id)
# applabels['label'] = labelencoder.transform(applabels.label_id)
# nlabels = len(labelencoder.classes_)
#==============================================================================


# New version
app_labels = app_labels.loc[app_labels.app_id.isin(appevents.app_id.unique())]
app_labels['app'] = appencoder.transform(app_labels.app_id)
labelencoder = LabelEncoder().fit(app_labels.general_groups)
app_labels['label'] = labelencoder.transform(app_labels.general_groups)
nlabels = len(labelencoder.classes_)

# In[ ]:

# del appevents
gatrain.head()


# In[ ]:

#devicelabels = (deviceapps[['device_id','app']]
#                .merge(app_labels[['app','label']])
#                .groupby(['device_id','label'])['app'].agg(['size'])
#                .merge(gatrain[['trainrow']], how='left',  left_index=True, right_index=True)
#                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
#               .reset_index())
devicelabels = pd.read_csv(os.path.join(datadir,'devicelabels2.csv'))
devicelabels.head()


# In[ ]:

d = devicelabels.dropna(subset=['trainrow'])
Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)), 
                      shape=(gatrain.shape[0],nlabels))
d = devicelabels.dropna(subset=['testrow'])
Xte_label = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.label)), 
                      shape=(gatest.shape[0],nlabels))
print('Labels data: train shape {}, test shape {}'.format(Xtr_label.shape, Xte_label.shape))


# # Join all features

Xtrain1 = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label, Xtr_loc, Xtr_period, Xtr_loc_act_m, Xtr_period_act_m, Xtr_loc_act, Xtr_period_act), format='csr')
Xtest1 =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label, Xte_loc, Xte_period, Xte_loc_act_m, Xte_period_act_m, Xte_loc_act, Xte_period_act), format='csr')

Xtrain2 = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label, Xtr_loc, Xtr_period), format='csr')
Xtest2 =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label, Xte_loc, Xte_period), format='csr')


Xtrain3 = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label, Xtr_loc, Xtr_period, Xtr_app_act_m, Xtr_period_act_m), format='csr')
Xtest3 =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label, Xte_loc, Xte_period, Xte_app_act_m, Xte_period_act_m), format='csr')


Xtrain4 = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label, Xtr_loc, Xtr_period, Xtr_app_act, Xtr_period_act), format='csr')
Xtest4 =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label, Xte_loc, Xte_period, Xte_app_act, Xte_period_act), format='csr')

Xtrain5 = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label, Xtr_loc, Xtr_period, Xtr_app_act_m, Xtr_loc_act_m), format='csr')
Xtest5 =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label, Xte_loc, Xte_period, Xte_app_act_m, Xte_loc_act_m), format='csr')

Xtrain6 = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label, Xtr_loc, Xtr_period, Xtr_period_act_m, Xtr_loc_act_m), format='csr')
Xtest6 =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label, Xte_loc, Xte_period, Xte_period_act_m, Xte_loc_act_m), format='csr')



print('All features: train shape {}, test shape {}'.format(Xtrain6.shape, Xtest6.shape))

targetencoder = LabelEncoder().fit(gatrain.group)
y = targetencoder.transform(gatrain.group)
nclasses = len(targetencoder.classes_)

#==============================================================================
# Keras Practice
#==============================================================================

def baseline_model1():
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=Xtrain1.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(80, input_dim=Xtrain1.shape[1], init='normal',activation='sigmoid'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(50, input_dim=Xtrain1.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return model
    
def baseline_model2():
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=Xtrain2.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(80, input_dim=Xtrain2.shape[1], init='normal',activation='sigmoid'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(50, input_dim=Xtrain2.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return model
    
def baseline_model3():
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=Xtrain3.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(80, input_dim=Xtrain3.shape[1], init='normal',activation='sigmoid'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(50, input_dim=Xtrain3.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return model
    
def baseline_model4():
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=Xtrain4.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(80, input_dim=Xtrain4.shape[1], init='normal',activation='sigmoid'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(50, input_dim=Xtrain4.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return model
    
def baseline_model5():
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=Xtrain5.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(80, input_dim=Xtrain5.shape[1], init='normal',activation='sigmoid'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(50, input_dim=Xtrain5.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return model
    
def baseline_model6():
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=Xtrain6.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(80, input_dim=Xtrain6.shape[1], init='normal',activation='sigmoid'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(50, input_dim=Xtrain6.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return model
    
#==============================================================================
# model=baseline_model()
# kf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=seed)
# pred = np.zeros((y.shape[0],nclasses))
# for itrain, itest in kf:
#         model=baseline_model()
#         print("Splitting Data")
#         Xtr, Xte = Xtrain[itrain, :], Xtrain[itest, :]
#         #Xtr_log, Xte_log = Xtrain_log[itrain, :], Xtrain_log[itest, :]
#         ytr, yte = y[itrain], y[itest]
#         print("Fitting Model")
#         fit = model.fit_generator(generator=batch_generator(Xtr,ytr,800,True),
#                                   nb_epoch=52,samples_per_epoch=Xtr.shape[0],verbose=2,validation_data=(Xte.todense(),yte))
#         print("Generating probabilities")
#         pred[itest,:]= model.predict_generator(generator=batch_generatorp(Xte, 800, False), val_samples=Xte.shape[0])
# 
#==============================================================================
params = {}
params['booster'] = 'gblinear'
params['objective'] = "multi:softprob"
params['eval_metric'] = 'mlogloss'
params['eta'] = 0.01
params['num_class'] = 12
params['lambda'] = 5
params['alpha'] = 3

parameters = {'C':0.02,'l1_ratio':1}

score(mix={'logit':1./12,'xgb':1./12,'sgd':0,'keras':1./12})

def score(parameters= {'C':0.02,'l1_ratio':0.5,'n_folds':5}, random_state = 0,mix={'logit':0.3,'xgb':0.3,'sgd':0,'keras':0.4}):
    kf = StratifiedKFold(y, n_folds=parameters['n_folds'], shuffle=True, random_state=random_state)
    pred = np.zeros((y.shape[0],nclasses))
    pred2 = np.zeros((y.shape[0],nclasses))
    pred_logistic1 = np.zeros((y.shape[0],nclasses))
    pred_xgb1 = np.zeros((y.shape[0],nclasses))
    pred_sgd1 = np.zeros((y.shape[0],nclasses))
    pred_keras1 = np.zeros((y.shape[0],nclasses))
    pred_logistic2 = np.zeros((y.shape[0],nclasses))
    pred_xgb2 = np.zeros((y.shape[0],nclasses))
    pred_sgd2 = np.zeros((y.shape[0],nclasses))
    pred_keras2 = np.zeros((y.shape[0],nclasses))
    pred_xgb3 = np.zeros((y.shape[0],nclasses))
    pred_keras3 = np.zeros((y.shape[0],nclasses))
    pred_xgb4 = np.zeros((y.shape[0],nclasses))
    pred_keras4 = np.zeros((y.shape[0],nclasses))
    pred_xgb5 = np.zeros((y.shape[0],nclasses))
    pred_keras5 = np.zeros((y.shape[0],nclasses))
    pred_xgb6 = np.zeros((y.shape[0],nclasses))
    pred_keras6 = np.zeros((y.shape[0],nclasses))
    logit_val = False
    xgb_val = False
    sgd_val = False
    keras_val = False
    if mix['logit']>0:
        logit_val = True
    if mix['xgb']>0:
        xgb_val = True
    if mix['sgd']>0:
        sgd_val = True
    if mix['keras']>0:
        keras_val = True
    for itrain, itest in kf:
        Xtr1, Xte1 = Xtrain1[itrain, :], Xtrain1[itest, :]
        Xtr2, Xte2 = Xtrain2[itrain, :], Xtrain2[itest, :]
        Xtr3, Xte3 = Xtrain3[itrain, :], Xtrain3[itest, :]
        Xtr4, Xte4 = Xtrain4[itrain, :], Xtrain4[itest, :]
        Xtr5, Xte5 = Xtrain5[itrain, :], Xtrain5[itest, :]
        Xtr6, Xte6 = Xtrain6[itrain, :], Xtrain6[itest, :]
        #Xtr_log, Xte_log = Xtrain_log[itrain, :], Xtrain_log[itest, :]
        ytr, yte = y[itrain], y[itest]
        if logit_val:
            print("Fitting Logistic Regression")
            # Logistic Regression
            print("Model 1")
            clf_logit1 = LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs')
            clf_logit1.fit(Xtr1, ytr)
            pred_logistic1[itest,:] = clf_logit1.predict_proba(Xte1)
            
            print("Model 2")
            clf_logit2 = LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs')
            clf_logit2.fit(Xtr2, ytr)
            pred_logistic2[itest,:] = clf_logit2.predict_proba(Xte2)
        if xgb_val:
            print("Fitting XGBoost")
            # XGBoost
            print("Model 1")
            d_train1 = xgb.DMatrix(Xtr1, label=ytr)
            d_valid1 = xgb.DMatrix(Xte1, label=yte)
            print(d_train1.num_col())
            print(d_valid1.num_col())
            watchlist = [(d_train1, 'train'), (d_valid1, 'eval')]
            clf_xgb1 = xgb.train(params, d_train1, 1000, watchlist, early_stopping_rounds=10,verbose_eval=False)
            print(clf_xgb1.best_iteration)
            pred_xgb1[itest,:] = clf_xgb1.predict(d_valid1)
            
            print("Model 2")
            d_train2 = xgb.DMatrix(Xtr2, label=ytr)
            d_valid2 = xgb.DMatrix(Xte2, label=yte)
            print(d_train2.num_col())
            print(d_valid2.num_col())
            watchlist = [(d_train2, 'train'), (d_valid2, 'eval')]
            clf_xgb2 = xgb.train(params, d_train2, 1000, watchlist, early_stopping_rounds=10,verbose_eval=False)
            print(clf_xgb2.best_iteration)
            pred_xgb2[itest,:] = clf_xgb2.predict(d_valid2)
            
            print("Model 3")
            d_train3 = xgb.DMatrix(Xtr3, label=ytr)
            d_valid3 = xgb.DMatrix(Xte3, label=yte)
            print(d_train3.num_col())
            print(d_valid3.num_col())
            watchlist = [(d_train3, 'train'), (d_valid3, 'eval')]
            clf_xgb3 = xgb.train(params, d_train3, 1000, watchlist, early_stopping_rounds=10,verbose_eval=False)
            print(clf_xgb3.best_iteration)
            pred_xgb3[itest,:] = clf_xgb3.predict(d_valid3)
            
            print("Model 4")
            d_train4 = xgb.DMatrix(Xtr4, label=ytr)
            d_valid4 = xgb.DMatrix(Xte4, label=yte)
            print(d_train4.num_col())
            print(d_valid4.num_col())
            watchlist = [(d_train4, 'train'), (d_valid4, 'eval')]
            clf_xgb4 = xgb.train(params, d_train4, 1000, watchlist, early_stopping_rounds=10,verbose_eval=False)#220
            print(clf_xgb4.best_iteration)
            pred_xgb4[itest,:] = clf_xgb4.predict(d_valid4)
            #print("Model 5")
            #d_train5 = xgb.DMatrix(Xtr5, label=ytr)
            #d_valid5 = xgb.DMatrix(Xte5, label=yte)
            #print(d_train5.num_col())
            #print(d_valid5.num_col())
            #watchlist = [(d_train5, 'train'), (d_valid5, 'eval')]
            #clf_xgb5 = xgb.train(params, d_train5, 1000, watchlist, early_stopping_rounds=10,verbose_eval=True)
            #pred_xgb5[itest,:] = clf_xgb5.predict(d_valid5)
            
            #print("Model 6")
            #d_train6 = xgb.DMatrix(Xtr6, label=ytr)
            #d_valid6 = xgb.DMatrix(Xte6, label=yte)
            #print(d_train6.num_col())
            #print(d_valid6.num_col())
            #watchlist = [(d_train6, 'train'), (d_valid6, 'eval')]
            #clf_xgb6 = xgb.train(params, d_train6, 1000, watchlist, early_stopping_rounds=10,verbose_eval=True)
            #pred_xgb6[itest,:] = clf_xgb6.predict(d_valid6)
        if sgd_val:
            # SGD Classifier
            print("Fitting SGD Classifier")
            clf3 = SGDClassifier(loss='log',penalty='elasticnet',l1_ratio=parameters['l1_ratio'])
            clf3.fit(Xtr1, ytr)
            pred_sgd1[itest,:] = clf3.predict_proba(Xte1)
        if keras_val:
            print("Fitting Neural Network")
            model1=baseline_model1()
            model2=baseline_model2()
            model3=baseline_model3()
            model4=baseline_model4()
            model5=baseline_model5()
            model6=baseline_model6()
            
            print("Model 1")
            fit = model1.fit_generator(generator=batch_generator(Xtr1,ytr,800,True),
                                  nb_epoch=50,samples_per_epoch=Xtr1.shape[0],verbose=2)
            pred_keras1[itest,:]= model1.predict_generator(generator=batch_generatorp(Xte1, 800, False), val_samples=Xte1.shape[0])
            
            print("Model 2")
            fit = model2.fit_generator(generator=batch_generator(Xtr2,ytr,800,True),
                                  nb_epoch=50,samples_per_epoch=Xtr2.shape[0],verbose=2)
            pred_keras2[itest,:]= model2.predict_generator(generator=batch_generatorp(Xte2, 800, False), val_samples=Xte2.shape[0])
            
            print("Model 3")
            fit = model3.fit_generator(generator=batch_generator(Xtr3,ytr,800,True),
                                  nb_epoch=50,samples_per_epoch=Xtr3.shape[0],verbose=2)
            pred_keras3[itest,:]= model3.predict_generator(generator=batch_generatorp(Xte3, 800, False), val_samples=Xte3.shape[0])
            
            print("Model 4")
            fit = model4.fit_generator(generator=batch_generator(Xtr4,ytr,800,True),
                                  nb_epoch=50,samples_per_epoch=Xtr4.shape[0],verbose=2)
            pred_keras4[itest,:]= model4.predict_generator(generator=batch_generatorp(Xte4, 800, False), val_samples=Xte4.shape[0])
            
            print("Model 5")
            fit = model5.fit_generator(generator=batch_generator(Xtr5,ytr,800,True),
                                  nb_epoch=50,samples_per_epoch=Xtr5.shape[0],verbose=2)
            pred_keras5[itest,:]= model5.predict_generator(generator=batch_generatorp(Xte5, 800, False), val_samples=Xte5.shape[0])
            
            print("Model 6")
            fit = model6.fit_generator(generator=batch_generator(Xtr6,ytr,800,True),
                                  nb_epoch=50,samples_per_epoch=Xtr6.shape[0],verbose=2)
            pred_keras6[itest,:]= model6.predict_generator(generator=batch_generatorp(Xte6, 800, False), val_samples=Xte6.shape[0])
        # Downsize to one fold only for kernels
        # Combine predictions
            
        tr_mat = (np.concatenate((clf_logit1.predict_proba(Xtr1),clf_logit2.predict_proba(Xtr2),clf_xgb1.predict(d_train1),clf_xgb2.predict(d_train2),
                clf_xgb3.predict(d_train3),clf_xgb4.predict(d_train4),model1.predict_generator(generator=batch_generatorp(Xtr1, 800, False), val_samples=Xtr1.shape[0]),
                model2.predict_generator(generator=batch_generatorp(Xtr2, 800, False), val_samples=Xtr2.shape[0]),
                model3.predict_generator(generator=batch_generatorp(Xtr3, 800, False), val_samples=Xtr3.shape[0]),
                model4.predict_generator(generator=batch_generatorp(Xtr4, 800, False), val_samples=Xtr4.shape[0]),
                model5.predict_generator(generator=batch_generatorp(Xtr5, 800, False), val_samples=Xtr5.shape[0]),
                model6.predict_generator(generator=batch_generatorp(Xtr6, 800, False), val_samples=Xtr6.shape[0])),axis=1))
                
        te_mat = (np.concatenate((pred_logistic1[itest,:],pred_logistic2[itest,:],pred_xgb1[itest,:],pred_xgb2[itest,:],
                pred_xgb3[itest,:],pred_xgb4[itest,:],pred_keras1[itest,:],
                pred_keras2[itest,:], pred_keras3[itest,:], pred_keras4[itest,:],
                pred_keras5[itest,:], pred_keras6[itest,:]),axis=1))
                
        pred[itest,:] = ((mix['logit']*pred_logistic1[itest,:])+(mix['logit']*pred_logistic2[itest,:])+
                        (mix['xgb']*pred_xgb1[itest,:])+(mix['xgb']*pred_xgb2[itest,:])+(mix['xgb']*pred_xgb3[itest,:])+(mix['xgb']*pred_xgb4[itest,:])+(0*mix['xgb']*pred_xgb5[itest,:])+(0*mix['xgb']*pred_xgb6[itest,:])+
                        (mix['sgd']*pred_sgd1[itest,:])+
                        (mix['keras']*pred_keras1[itest,:])+(mix['keras']*pred_keras2[itest,:])+(mix['keras']*pred_keras3[itest,:])+(mix['keras']*pred_keras4[itest,:])+(mix['keras']*pred_keras5[itest,:])+(mix['keras']*pred_keras6[itest,:]))
                        
        d_train = xgb.DMatrix(tr_mat,label=ytr)
        d_valid = xgb.DMatrix(te_mat,label=yte)
        watchlist = [(d_train, 'train'), (d_valid, 'eval')]
        clf = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=10,verbose_eval=False)
        pred2[itest,:] = clf.predict(d_valid)
        #return log_loss(yte, pred[itest, :])
        print("Logistic1 {:.5f}, Logistic2 {:.5f}".format(log_loss(yte, pred_logistic1[itest,:]),log_loss(yte, pred_logistic2[itest,:]), end=' '))
        print('')
        print("XGB1 {:.5f}, XGB2 {:.5f}, XGB3 {:.5f}, XGB4 {:.5f}, XGB5 {:.5f}, XGB6 {:.5f}".format(log_loss(yte, pred_xgb1[itest,:]),log_loss(yte, pred_xgb2[itest,:]),log_loss(yte, pred_xgb3[itest,:]),log_loss(yte, pred_xgb4[itest,:]),log_loss(yte, pred_xgb5[itest,:]),log_loss(yte, pred_xgb6[itest,:]), end=' '))
        print('')
        print("NEURAL1 {:.5f}, NEURAL2 {:.5f}, NEURAL3 {:.5f}, NEURAL4 {:.5f}, NEURAL5 {:.5f}, NEURAL6 {:.5f}".format(log_loss(yte, pred_keras1[itest,:]),log_loss(yte, pred_keras2[itest,:]),log_loss(yte, pred_keras3[itest,:]),log_loss(yte, pred_keras4[itest,:]),log_loss(yte, pred_keras5[itest,:]),log_loss(yte, pred_keras6[itest,:]), end=' '))
        print('')
        print("Average Mix {:.5f}, Average XGB {:.5f}".format(log_loss(yte,pred[itest,]),log_loss(yte,pred2[itest,]),end=' '))
        print('')
    print('')
    return log_loss(y, pred)

###############################################################################
# Submitted model
###############################################################################


#pred = np.zeros((y.shape[0],nclasses))
#pred_logistic1 = np.zeros((y.shape[0],nclasses))
#pred_xgb1 = np.zeros((y.shape[0],nclasses))
#pred_sgd1 = np.zeros((y.shape[0],nclasses))
#pred_keras1 = np.zeros((y.shape[0],nclasses))
#pred_logistic2 = np.zeros((y.shape[0],nclasses))
#pred_xgb2 = np.zeros((y.shape[0],nclasses))
#pred_sgd2 = np.zeros((y.shape[0],nclasses))
#pred_keras2 = np.zeros((y.shape[0],nclasses))
#pred_xgb3 = np.zeros((y.shape[0],nclasses))
#pred_keras3 = np.zeros((y.shape[0],nclasses))
#pred_xgb4 = np.zeros((y.shape[0],nclasses))
#pred_keras4 = np.zeros((y.shape[0],nclasses))
#pred_keras5 = np.zeros((y.shape[0],nclasses))
#pred_keras6 = np.zeros((y.shape[0],nclasses))

        #Xtr_log, Xte_log = Xtrain_log[itrain, :], Xtrain_log[itest, :]
# Logistic Regression
print("Model 1")
clf_logit1 = LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs')
clf_logit1.fit(Xtrain1, y)
pred_logistic1 = clf_logit1.predict_proba(Xtest1)

pred_logistic11 = pd.DataFrame(pred_logistic1, index = gatest.index, columns=targetencoder.classes_)
pred_logistic11.to_csv('pred_logistic11.csv',index=True)

print("Model 2")
clf_logit2 = LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs')
clf_logit2.fit(Xtrain2, y)
pred_logistic2 = clf_logit2.predict_proba(Xtest2)

pred_logistic22 = pd.DataFrame(pred_logistic2, index = gatest.index, columns=targetencoder.classes_)
pred_logistic22.to_csv('pred_logistic22.csv',index=True)

# XGBoost
print("Model 1")
d_train = xgb.DMatrix(Xtrain1, label=y)
d_valid = xgb.DMatrix(Xtest1)

watchlist = [(d_train, 'train')]
clf_xgb1 = xgb.train(params, d_train, 300, watchlist, early_stopping_rounds=10,verbose_eval=True)
pred_xgb1 = clf_xgb1.predict(d_valid)

pred_xgb11 = pd.DataFrame(pred_xgb1, index = gatest.index, columns=targetencoder.classes_)
pred_xgb11.to_csv('pred_xgb11.csv',index=True)

print("Model 2")
d_train = xgb.DMatrix(Xtrain2, label=y)
d_valid = xgb.DMatrix(Xtest2)

watchlist = [(d_train, 'train')]
clf_xgb2 = xgb.train(params, d_train, 300, watchlist, early_stopping_rounds=10,verbose_eval=True)
pred_xgb2 = clf_xgb2.predict(d_valid)

pred_xgb22 = pd.DataFrame(pred_xgb2, index = gatest.index, columns=targetencoder.classes_)
pred_xgb22.to_csv('pred_xgb22.csv',index=True)


print("Model 3")
d_train = xgb.DMatrix(Xtrain3, label=y)
d_valid= xgb.DMatrix(Xtest3)

watchlist = [(d_train, 'train')]
clf_xgb3 = xgb.train(params, d_train, 300, watchlist, early_stopping_rounds=10,verbose_eval=True)
pred_xgb3 = clf_xgb3.predict(d_valid)

pred_xgb33 = pd.DataFrame(pred_xgb3, index = gatest.index, columns=targetencoder.classes_)
pred_xgb33.to_csv('pred_xgb33.csv',index=True)


print("Model 4")
d_train = xgb.DMatrix(Xtrain4, label=y)
d_valid = xgb.DMatrix(Xtest4)

watchlist = [(d_train, 'train')]
clf_xgb4 = xgb.train(params, d_train, 220, watchlist, early_stopping_rounds=10,verbose_eval=True)#220
pred_xgb4 = clf_xgb4.predict(d_valid)

pred_xgb44 = pd.DataFrame(pred_xgb4, index = gatest.index, columns=targetencoder.classes_)
pred_xgb44.to_csv('pred_xgb44.csv',index=True)


#print("Model 5")
#d_train5 = xgb.DMatrix(Xtr5, label=ytr)
#d_valid5 = xgb.DMatrix(Xte5, label=yte)
#print(d_train5.num_col())
#print(d_valid5.num_col())
#watchlist = [(d_train5, 'train'), (d_valid5, 'eval')]
#clf_xgb5 = xgb.train(params, d_train5, 1000, watchlist, early_stopping_rounds=10,verbose_eval=True)
#pred_xgb5[itest,:] = clf_xgb5.predict(d_valid5)

#print("Model 6")
#d_train6 = xgb.DMatrix(Xtr6, label=ytr)
#d_valid6 = xgb.DMatrix(Xte6, label=yte)
#print(d_train6.num_col())
#print(d_valid6.num_col())
#watchlist = [(d_train6, 'train'), (d_valid6, 'eval')]
#clf_xgb6 = xgb.train(params, d_train6, 1000, watchlist, early_stopping_rounds=10,verbose_eval=True)
#pred_xgb6[itest,:] = clf_xgb6.predict(d_valid6)

# Keras Neural Network

model1=baseline_model1()
model2=baseline_model2()
model3=baseline_model3()
model4=baseline_model4()
model5=baseline_model5()
model6=baseline_model6()

print("Model 1")
fit = model1.fit_generator(generator=batch_generator(Xtrain1,y,800,True),
                      nb_epoch=50,samples_per_epoch=Xtrain1.shape[0],verbose=2)
pred_keras1 = model1.predict_generator(generator=batch_generatorp(Xtest1, 800, False), val_samples=Xtest1.shape[0])

pred_keras11 = pd.DataFrame(pred_keras1, index = gatest.index, columns=targetencoder.classes_)
pred_keras11.to_csv('pred_keras11.csv',index=True)


print("Model 2")
fit = model2.fit_generator(generator=batch_generator(Xtrain2,y,800,True),
                      nb_epoch=50,samples_per_epoch=Xtrain2.shape[0],verbose=2)
pred_keras2 = model2.predict_generator(generator=batch_generatorp(Xtest2, 800, False), val_samples=Xtest2.shape[0])

pred_keras22 = pd.DataFrame(pred_keras2, index = gatest.index, columns=targetencoder.classes_)
pred_keras22.to_csv('pred_keras22.csv',index=True)

print("Model 3")
fit = model3.fit_generator(generator=batch_generator(Xtrain3,y,800,True),
                      nb_epoch=50,samples_per_epoch=Xtrain3.shape[0],verbose=2)
pred_keras3 = model3.predict_generator(generator=batch_generatorp(Xtest3, 800, False), val_samples=Xtest3.shape[0])

pred_keras33 = pd.DataFrame(pred_keras3, index = gatest.index, columns=targetencoder.classes_)
pred_keras33.to_csv('pred_keras33.csv',index=True)

print("Model 4")
fit = model4.fit_generator(generator=batch_generator(Xtrain4,y,800,True),
                      nb_epoch=50,samples_per_epoch=Xtrain4.shape[0],verbose=2)
pred_keras4 = model4.predict_generator(generator=batch_generatorp(Xtest4, 800, False), val_samples=Xtest4.shape[0])

pred_keras44 = pd.DataFrame(pred_keras4, index = gatest.index, columns=targetencoder.classes_)
pred_keras44.to_csv('pred_keras44.csv',index=True)

print("Model 5")
fit = model5.fit_generator(generator=batch_generator(Xtrain5,y,800,True),
                      nb_epoch=50,samples_per_epoch=Xtrain5.shape[0],verbose=2)
pred_keras5 = model5.predict_generator(generator=batch_generatorp(Xtest5, 800, False), val_samples=Xtest5.shape[0])

pred_keras55 = pd.DataFrame(pred_keras5, index = gatest.index, columns=targetencoder.classes_)
pred_keras55.to_csv('pred_keras55.csv',index=True)

print("Model 6")
fit = model6.fit_generator(generator=batch_generator(Xtrain6,y,800,True),
                      nb_epoch=50,samples_per_epoch=Xtrain6.shape[0],verbose=2)
pred_keras6 = model6.predict_generator(generator=batch_generatorp(Xtest6, 800, False), val_samples=Xtest6.shape[0])

pred_keras66 = pd.DataFrame(pred_keras6, index = gatest.index, columns=targetencoder.classes_)
pred_keras66.to_csv('pred_keras66.csv',index=True)

# Downsize to one fold only for kernels
# Combine predictions
mix = {'logit':1./12,'xgb':1./12,'keras':1./12}

pred = ((mix['logit']*pred_logistic1)+(mix['logit']*pred_logistic2)+
            (mix['xgb']*pred_xgb1)+(mix['xgb']*pred_xgb2)+(mix['xgb']*pred_xgb3)+(mix['xgb']*pred_xgb4)+
            (mix['keras']*pred_keras1)+(mix['keras']*pred_keras2)+(mix['keras']*pred_keras3)+(mix['keras']*pred_keras4)+(mix['keras']*pred_keras5)+(mix['keras']*pred_keras6))



pred_pd = pd.DataFrame(pred, index = gatest.index, columns=targetencoder.classes_)
pred_pd.to_csv('pred_ensemble12.csv',index=True)







###############################################################################









clf_logit = LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs')
clf_logit.fit(Xtrain,y)
pred1 = clf_logit.predict_proba(Xtest)


d_train = xgb.DMatrix(Xtrain, label=y)
d_valid = xgb.DMatrix(Xtest)
watchlist = [(d_train, 'train')]
clf_xgb = xgb.train(params, d_train, 300, watchlist, early_stopping_rounds=10,verbose_eval=True)
pred2 = clf_xgb.predict(d_valid)

clf_keras = baseline_model()
clf_keras.fit_generator(generator=batch_generator(Xtrain,y,800,True),
                                  nb_epoch=50,samples_per_epoch=Xtrain.shape[0],verbose=2)
pred3= clf_keras.predict_generator(generator=batch_generatorp(Xtest, 800, False), val_samples=Xtest.shape[0])

pred = pred1*(1./3) + pred2*(1./3) + pred3*(1./3)
pred = pd.DataFrame(pred, index = gatest.index, columns=targetencoder.classes_)
pred.head()

pred.to_csv('logit_xgb_keras_subm.csv',index=True)

