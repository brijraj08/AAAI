import sys 
sys.path.append("/usr/lib/python3/dist-packages")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from util.data_utils import create_seq_db_filter_top_k1, sequences_to_spfm_format
from util.split import last_session_out_split
from util.metrics import precision, recall, mrr
from util import evaluation
import pandas as pd
from recommenders.RNNRecommender import RNNRecommender


import datetime

def get_test_sequences_and_users(test_data, given_k, train_users):
    # we can run evaluation only over sequences longer than abs(LAST_K)
    mask = test_data['sequence'].map(len) > abs(given_k)
    mask &= test_data['user_id'].isin(train_users)
    test_sequences = test_data.loc[mask, 'sequence'].values
    test_users = test_data.loc[mask, 'user_id'].values
    return test_sequences, test_users


# 1. Load the dataset
ratings=pd.read_csv("lastfm_data_fynl1.csv")


#sessions with 15mins interval 
def make_sessions(data, session_th=900, is_ordered=False, user_key='user_id', item_key='item_id', time_key='ts'):
    """Assigns session ids to the events in data without grouping keys"""
    if not is_ordered:
        # sort data by user and time
        data.sort_values(by=[user_key, time_key], ascending=True, inplace=True)
    # compute the time difference between queries
    tdiff = np.diff(data[time_key].values)
    # check which of them are bigger then session_th
    split_session = tdiff > session_th
    split_session = np.r_[True, split_session]
    # check when the user chenges is data
    new_user = data[user_key].values[1:]!= data[user_key].values[:-1]
    new_user = np.r_[True, new_user]
    # a new sessions stars when at least one of the two conditions is verified
    new_session = np.logical_or(new_user, split_session)
    # compute the session idsi
    session_ids = np.cumsum(new_session)
    data['session_id'] = session_ids
    return data

# %%
sessions_df = make_sessions(ratings)

from tqdm import tqdm

arr = []
i = 0
user_ = []
session_ = []
ts_ = []
it = []
pbar = tqdm(total=len(sessions_df)-1)
while i < len(sessions_df):
    #ts_.append(sessions_df['ts'][sessions_df.index[i]])
    session_id = sessions_df['session_id'][sessions_df.index[i]]
    it.append(sessions_df['item_id'][sessions_df.index[i]])
    if i == len(sessions_df) - 1 or \
       session_id != sessions_df['session_id'][sessions_df.index[i+1]]:
        user_.append(sessions_df['user_id'][sessions_df.index[i]])
        session_.append(session_id)
        arr.append(it)
        it = []
        ts_.append(sessions_df['ts'][sessions_df.index[i]])
    i += 1
    pbar.update(1)
#pbar.close()


new_dataset= pd.DataFrame()
new_dataset['user_id']=pd.Series(user_)
new_dataset['session_id']=pd.Series(session_)
new_dataset['sequence']=pd.Series(arr)
new_dataset['ts']=pd.Series(ts_)

dataset=new_dataset

def process_sequences(group):
    if len(group['sequence'].iloc[-1]) == 1 and len(group) > 1:
        group['sequence'].iloc[-2].extend(group['sequence'].iloc[-1])
        return group.iloc[:-1]  
    return group

dataset = dataset.groupby('user_id').apply(process_sequences).reset_index(drop=True)
def process_user(user_df):    
    while len(user_df) > 1:
        pre_sq_list = []
        for seq in user_df.iloc[:-1]['sequence']:
            pre_sq_list.extend(seq)  
        pre_sq = set(pre_sq_list) 
        last_sqitems = user_df.iloc[-1]['sequence']
        #last_sqitems_set = set(last_sqitems)
        #common_last_seq = list(last_sqitems_set.intersection(pre_sq))
        common_last_seq =[item for item in last_sqitems if item in pre_sq]
        if len(common_last_seq) > 1:
            user_df.at[user_df.index[-1], 'sequence'] = common_last_seq
            break  
        elif len(common_last_seq) == 1:
            if len(user_df) > 1:  
                user_df.iloc[-2]['sequence'].extend(common_last_seq)
            user_df = user_df.iloc[:-1]  
        else:
            user_df = user_df.iloc[:-1]  
    return user_df
dataset=dataset.groupby('user_id').apply(process_user).reset_index(drop=True)

#Let's show some statistics about the dataset
from collections import Counter
cnt = Counter()
dataset.sequence.map(cnt.update);

#dataset = dataset[dataset['sequence'].map(len) > 4]
sequence_length = dataset.sequence.map(len).values

n_sessions_per_user = dataset.groupby('user_id').size()

print('Number of items: {}'.format(len(cnt)))
print('Number of users: {}'.format(dataset.user_id.nunique()))
print('Number of sessions: {}'.format(len(dataset)) )

print('\nSession length:\n\tAverage: {:.2f}\n\tMedian: {}\n\tMin: {}\n\tMax: {}'.format(
    sequence_length.mean(), 
    np.quantile(sequence_length, 0.5), 
    sequence_length.min(), 
    sequence_length.max()))

print('Sessions per user:\n\tAverage: {:.2f}\n\tMedian: {}\n\tMin: {}\n\tMax: {}'.format(
    n_sessions_per_user.mean(), 
    np.quantile(n_sessions_per_user, 0.5), 
    n_sessions_per_user.min(), 
    n_sessions_per_user.max()))



# 2. Split the dataset
#For simplicity, let's split the dataset by assigning the **last session** of every user to the **test set**, and **all the previous** ones to the **training set**.


train_data, test_data = last_session_out_split(dataset)
print("Train sessions: {} - Test sessions: {}".format(len(train_data), len(test_data)))

recommender = RNNRecommender(session_layers=[100], 
                             user_layers=[100],
                             batch_size=16,#150,
                             learning_rate=0.1,
                             momentum=0.0,
                             dropout=(0.0,0.1,0.0),
                             epochs=80,
                             personalized=True)

recommender.fit(train_data)

METRICS = {'precision':precision, 
            'recall':recall,
            'mrr': mrr}
TOPN = 20 # length of the recommendation list
    
        
            # GIVEN_K=1, LOOK_AHEAD=1, STEP=1 corresponds to the classical next-item evaluation
GIVEN_K = 1
LOOK_AHEAD = 1
STEP = 1
                

test_sequences, test_users = get_test_sequences_and_users(test_data, GIVEN_K, train_data['user_id'].values) # we need user ids now!
print('{} sequences available for evaluation ({} users)'.format(len(test_sequences), len(np.unique(test_users))))

results = evaluation.sequential_evaluation(recommender,
                                               test_sequences=test_sequences,#[:500] ,
                                               users=test_users,#[:500],
                                               given_k=GIVEN_K,
                                               look_ahead=LOOK_AHEAD,
                                               evaluation_functions=METRICS.values(),
                                               top_n=TOPN,
                                               scroll=True,  # scrolling averages metrics over all profile lengths
                                               step=STEP)


print('Sequential evaluation (GIVEN_K={}, LOOK_AHEAD={}, STEP={})'.format(GIVEN_K, LOOK_AHEAD, STEP))
for mname, mvalue in zip(METRICS.keys(), results):
    print('\t{}@{}: {:.4f}'.format(mname, TOPN, mvalue))



METRICS = {'precision':precision,
            'recall':recall,
            'mrr': mrr}
TOPN = 10 # length of the recommendation list


            # GIVEN_K=1, LOOK_AHEAD=1, STEP=1 corresponds to the classical next-item evaluation
GIVEN_K = 1
LOOK_AHEAD = 1
STEP = 1


test_sequences, test_users = get_test_sequences_and_users(test_data, GIVEN_K, train_data['user_id'].values) # we need user ids now!
print('{} sequences available for evaluation ({} users)'.format(len(test_sequences), len(np.unique(test_users))))

results = evaluation.sequential_evaluation(recommender,
                                               test_sequences=test_sequences,#[:500] ,
                                               users=test_users,#[:500],
                                               given_k=GIVEN_K,
                                               look_ahead=LOOK_AHEAD,
                                               evaluation_functions=METRICS.values(),
                                               top_n=TOPN,
                                               scroll=True,  # scrolling averages metrics over all profile lengths
                                               step=STEP)


print('Sequential evaluation (GIVEN_K={}, LOOK_AHEAD={}, STEP={})'.format(GIVEN_K, LOOK_AHEAD, STEP))
for mname, mvalue in zip(METRICS.keys(), results):
    print('\t{}@{}: {:.4f}'.format(mname, TOPN, mvalue))

