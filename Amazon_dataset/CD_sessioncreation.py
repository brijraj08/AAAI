import numpy as np
import pandas as pd
from tqdm import tqdm
def process_item_set(item_set, item_set_id):
    win=np.zeros(len(item_set))
    ses_index=[]
    hash_array=np.zeros(141)
    marker=0
    i=0
    while i < len(item_set):
        x=item_set[i]
        win[i]=x
        hash_array[int(x)]=hash_array[int(x)]+1
        if hash_array[int(x)]>2 and sum([int(i) for i in hash_array>2])<2:#len(np.argwhere(hash_array==np.max(hash_array)))<=1:
                marker=i
        if np.max(hash_array) > 2 and   sum([int(i) for i in hash_array>2])==2:        #len(np.argwhere(hash_array==np.max(hash_array)))>1:
            ses_index.append(marker)
            if marker+1 < len(item_set)-1:
                i=marker
            marker=0
            hash_array=np.zeros(141)
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
data = pd.read_csv("")
data.sort_values(by=['user_id', 'ts'], ascending=True, inplace=True)
grouped_data = data.groupby("user_id")

results = []
results_item_id = []
session_count = 0

for user_id, group in tqdm(grouped_data):
    item_set = group['cluster_id'].to_numpy()
    item_set_id = group['item_id'].to_numpy()
    sequences, sequences_item_id = process_item_set(item_set, item_set_id)
    for seq, seq_item_id in zip(sequences, sequences_item_id):
        session_count += 1  
        results.append({
            'user_id': user_id,
            'sequence': seq,
            'ts': group['ts'].min(),
            'session_id': session_count
        })
        results_item_id.append({
            'user_id': user_id,
            'sequence': seq_item_id,
            'ts':group['ts'].min(),
            'session_id': session_count
        })

result_data = pd.DataFrame(results)
result_data_item_id = pd.DataFrame(results_item_id)

df=result_data_item_id
df['sequence'] = df['sequence'].astype(str).str.replace('[\[\]]', '').str.split().apply(lambda x: [int(i.strip()) for i in x])

df_exploded = df.explode('sequence')

df_exploded = df_exploded.rename(columns={'sequence': 'item_id', 'ts': 'Time', 'session_id': 'SessionId'})

df_exploded = df_exploded.merge(data[['user_id', 'item_id', 'ts']], on=['user_id', 'item_id'], how='left')
df_exploded = df_exploded.drop('Time', axis=1).rename(columns={'ts': 'Time'})
df_exploded.to_csv(r'', index=False)

