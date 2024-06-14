# -*- coding: utf-8 -*-
"""
@author: 7000030999
"""
from tqdm import tqdm
import sys 
sys.path.append("/usr/lib/python3/dist-packages")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.data_utils import create_seq_db_filter_top_k1, sequences_to_spfm_format
from util.split import last_session_out_split
from util.metrics import precision, recall, mrr
from util import evaluation1 as evaluation
import pandas as pd
from recommenders.RNNRecommender import RNNRecommender
import datetime
from tqdm import tqdm

def get_test_sequences_and_users(test_data, given_k, train_users):
    # we can run evaluation only over sequences longer than abs(LAST_K)
    mask = test_data['sequence'].map(len) > abs(given_k)
    mask &= test_data['user_id'].isin(train_users)
    test_sequences = test_data.loc[mask, 'sequence'].values
    test_users = test_data.loc[mask, 'user_id'].values
    return test_sequences, test_users

def process_item_set(item_set, item_set_id):
    win=np.zeros(len(item_set))
    ses_index=[]
    hash_array=np.zeros(63)
    marker=0
    i=0
    while i < len(item_set):
        x=int(item_set[i])
        win[i]=x
        hash_array[int(x)]=hash_array[int(x)]+1
        if hash_array[x]>2 and sum([int(i) for i in hash_array>2])<2:#len(np.argwhere(hash_array==np.max(hash_array)))<=1:
                marker=i
        if np.max(hash_array) > 2 and   sum([int(i) for i in hash_array>2])==2:        #len(np.argwhere(hash_array==np.max(hash_array)))>1:
            ses_index.append(marker)
            if marker+1 < len(item_set)-1:
                i=marker
            marker=0
            hash_array=np.zeros(63)
        i=i+1
    ses = []
    ses_item_id = []

    if ses_index:  # Check if ses_index is not empty
        ses.append(item_set[0:ses_index[0] + 1])
        ses_item_id.append(item_set_id[0:ses_index[0] + 1])
        for i in range(0, len(ses_index)):
            if i < len(ses_index) - 1:
                ses.append(item_set[ses_index[i] + 1: ses_index[i + 1] + 1])
                ses_item_id.append(item_set_id[ses_index[i] + 1: ses_index[i + 1] + 1])
        ses.append(item_set[ses_index[-1] + 1:])
        ses_item_id.append(item_set_id[ses_index[-1] + 1:])
    else:  # If ses_index is empty, add the entire item_set to ses and ses_item_id
        ses.append(item_set)
        ses_item_id.append(item_set_id)
    return ses, ses_item_id    
"""
    ses=[]
    ses_item_id=[]
    ses.append(item_set[0:ses_index[0]+1])
    for i in range(0, len(ses_index)):
        if i< len(ses_index)-1:
            ses.append(item_set[ses_index[i]+1: ses_index[i+1]+1])
            ses_item_id.append(item_set_id[ses_index[i]+1: ses_index[i+1]+1])
    ses.append(item_set[ses_index[-1]+1:])
    ses_item_id.append(item_set_id[ses_index[-1]+1:])

    return ses, ses_item_id
"""
# Read the CSV file
# data = pd.read_csv("updated_ratingwithcluster_id.csv")
# data = pd.read_csv("/Users/7000028834/Documents/Product_Recommendation/amazon_data.csv").head()
# data1 = pd.read_csv("/Users/7000028834/Documents/Product_Recommendation/amazon_data_12_jan_1.csv").head(10000)

data = pd.read_csv("/home/pankaj/brijraj1/27_jan_data.csv")
data=data.head(50000)
# Group the data by user_id
grouped_data = data.groupby("user_id")
# grouped_data1 = data1.groupby("user_id")
# Initialize an empty DataFrame to store the results
result_data = pd.DataFrame(columns=['user_id', 'sequence', 'ts', 'session_id'])


result_data_item_id = pd.DataFrame(columns=['user_id', 'sequence', 'ts', 'session_id'])

session_count = 0  # Initialize the session counter outside the user loop

# Process each group
count=0
count1=0
for user_id, group in tqdm(grouped_data):
    if len(group)>5:
        count=count+1
        item_set = group['cluster_id'].to_numpy()
        item_set_id = group['item_id'].to_numpy()
        sequences, sequences_item_id = process_item_set(item_set, item_set_id)
    
        for seq, seq_item_id in zip(sequences, sequences_item_id):
            session_count += 1  # Increment the session counter
    
            new_row = pd.DataFrame({
                'user_id': [user_id],
                'sequence': [seq],
                'ts': [group['ts'].min()],
                'session_id': [session_count]
            })
            result_data = pd.concat([result_data, new_row], ignore_index=True)
    
            new_row_item_id = pd.DataFrame({
                'user_id': [user_id],
                'sequence': [seq_item_id],
                'ts': [group['ts'].min()],
                'session_id': [session_count]
            })
            result_data_item_id = pd.concat([result_data_item_id, new_row_item_id], ignore_index=True)
    else:
        count1=count1+1
    #if count==1:
       # break


dataset=result_data_item_id


from collections import Counter
cnt = Counter()
dataset.sequence.map(cnt.update);

# %%
sequence_length = dataset.sequence.map(len).values
#dataset = dataset[dataset.sequence.map(len) > 2]
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

# 2. train and test the dataset
# %%
train_data, test_data = last_session_out_split(dataset)
print("Train sessions: {} - Test sessions: {}".format(len(train_data), len(test_data)))

# %%

recommender = RNNRecommender(session_layers=[600], 
                             user_layers=[600],
                             batch_size=256,
                             learning_rate=0.06,
                             momentum=0.01,
                             dropout=(0.2,0.2,0.2),
                             epochs=10,
                             personalized=True)
recommender.fit(train_data)


# %%
METRICS = {'precision':precision, 
            'recall':recall,
            'mrr': mrr}
TOPN = 5 # length of the recommendation list
    
        # %%
            # GIVEN_K=1, LOOK_AHEAD=1, STEP=1 corresponds to the classical next-item evaluation
GIVEN_K = 1
LOOK_AHEAD = 1
STEP = 1
                
# %%
test_sequences, test_users = get_test_sequences_and_users(test_data, GIVEN_K, train_data['user_id'].values) # we need user ids now!
print('{} sequences available for evaluation ({} users)'.format(len(test_sequences), len(np.unique(test_users))))

results = evaluation.sequential_evaluation(recommender,
                                               test_sequences=test_sequences,
                                               users=test_users,
                                               given_k=GIVEN_K,
                                               look_ahead=LOOK_AHEAD,
                                               evaluation_functions=METRICS.values(),
                                               top_n=TOPN,
                                               scroll=True,  # scrolling averages metrics over all profile lengths
                                               step=STEP)

# %%
print('Sequential evaluation (GIVEN_K={}, LOOK_AHEAD={}, STEP={})'.format(GIVEN_K, LOOK_AHEAD, STEP))
for mname, mvalue in zip(METRICS.keys(), results[0]):
    print('\t{}@{}: {:.4f}'.format(mname, TOPN, mvalue))

METRICS = {'precision':precision, 
           'recall':recall,
           'mrr': mrr}
TOPN = 10
GIVEN_K = 1
LOOK_AHEAD = 1
STEP = 1

# %%
# test_sequences, test_users = get_test_sequences_and_users(test_data, GIVEN_K, train_data['user_id'].values) # we need user ids now!
# print('{} sequences available for evaluation ({} users)'.format(len(test_sequences), len(np.unique(test_users))))


# results = evaluation.sequential_evaluation(recommender,
#                                            test_sequences=test_sequences,
#                                            users=test_users,
#                                            given_k=GIVEN_K,
#                                            look_ahead=LOOK_AHEAD,
#                                            evaluation_functions=METRICS.values(),
#                                            top_n=TOPN,
#                                            scroll=True,  # scrolling averages metrics over all profile lengths
#                                            step=STEP)
print('Sequential evaluation (GIVEN_K={}, LOOK_AHEAD={}, STEP={})'.format(GIVEN_K, LOOK_AHEAD, STEP))
for mname, mvalue in zip(METRICS.keys(), results[1]):
    print('\t{}@{}: {:.4f}'.format(mname, TOPN, mvalue))


METRICS = {'precision':precision, 
           'recall':recall,
           'mrr': mrr}
TOPN = 20 # length of the recommendation list
# %%
# GIVEN_K=1, LOOK_AHEAD=1, STEP=1 corresponds to the classical next-item evaluation
GIVEN_K = 1
LOOK_AHEAD = 1
STEP = 1

# %%
# test_sequences, test_users = get_test_sequences_and_users(test_data, GIVEN_K, train_data['user_id'].values) # we need user ids now!
# print('{} sequences available for evaluation ({} users)'.format(len(test_sequences), len(np.unique(test_users))))

# results = evaluation.sequential_evaluation(recommender,
#                                            test_sequences=test_sequences,
#                                            users=test_users,
#                                            given_k=GIVEN_K,
#                                            look_ahead=LOOK_AHEAD,
#                                            evaluation_functions=METRICS.values(),
#                                            top_n=TOPN,
#                                            scroll=True,  # scrolling averages metrics over all profile lengths
#                                            step=STEP)


# %%
print('Sequential evaluation (GIVEN_K={}, LOOK_AHEAD={}, STEP={})'.format(GIVEN_K, LOOK_AHEAD, STEP))
for mname, mvalue in zip(METRICS.keys(), results[2]):
    print('\t{}@{}: {:.4f}'.format(mname, TOPN, mvalue))
