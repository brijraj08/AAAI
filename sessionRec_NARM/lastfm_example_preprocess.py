import time
import csv
import pickle
from datetime import datetime
import operator
import pandas as pd

# Load .csv dataset
with open("lastfm_tssess900.csv", "r") as f:
#with open("lastfm_homosess_final1.csv", "r") as f:
    reader = csv.DictReader(f, delimiter=',')
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = data['SessionId']
        if curdate and not curid == sessid:
            #date = pd.to_datetime(curdate).timestamp()     

            date = time.mktime(time.strptime(curdate, '%Y-%m-%d %H:%M:%S%z'))
            #date = time.mktime(time.strptime(curdate, '%d-%m-%Y'))
            sess_date[curid] = date
        curid = sessid
        item = data['ItemId']
        curdate = data['Time']
        if sessid in sess_clicks:

            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
        if ctr % 100000 == 0:
            print ('Loaded', ctr)
    #date = pd.to_datetime(curdate).timestamp()  #bookdata

    date = time.mktime(time.strptime(curdate, '%Y-%m-%d %H:%M:%S%z'))
    #date = time.mktime(time.strptime(curdate, '%d-%m-%Y'))
    sess_date[curid] = date
"""
# Filter out length 1 sessions
for s in sess_clicks.keys():
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]
"""
for s in list(sess_clicks.keys()):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]


# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:

            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

for s in list(sess_clicks.keys()):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

"""
for s in sess_clicks.keys():
    curseq = sess_clicks[s]
    filseq = filter(lambda i: iid_counts[i] >= 5, curseq)
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# Split out test set based on dates
dates = sess_date.items()
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date
"""
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date


splitdate = maxdate - 86400 * 90  # 3 months for lastfm dataset 

print('Split date', splitdate)
train_sess = filter(lambda x: x[1] < splitdate, dates)
test_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
train_sess = sorted(train_sess, key=operator.itemgetter(1))
test_sess = sorted(test_sess, key=operator.itemgetter(1))

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
item_ctr = 1
train_seqs = []
train_dates = []
# Convert training sessions to sequences and renumber items to start from 1
for s, date in train_sess:
    seq = sess_clicks[s]
    outseq = []
    for i in seq:
        if i in item_dict:

            outseq += [item_dict[i]]
        else:
            outseq += [item_ctr]
            item_dict[i] = item_ctr
            item_ctr += 1
    if len(outseq) < 2:  # Doesn't occur
        continue
    train_seqs += [outseq]
    train_dates += [date]

test_seqs = []
test_dates = []
# Convert test sessions to sequences, ignoring items that do not appear in training set

for s, date in test_sess:
    seq = sess_clicks[s]
    outseq = []
    for i in seq:
        if i in item_dict:
            outseq += [item_dict[i]]
    if len(outseq) < 2:
        continue
    test_seqs += [outseq]
    test_dates += [date]

print(item_ctr)

def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    for seq, date in zip(iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]

    return out_seqs, out_dates, labs


tr_seqs, tr_dates, tr_labs = process_seqs(train_seqs,train_dates)
te_seqs, te_dates, te_labs = process_seqs(test_seqs,test_dates)

train = (tr_seqs, tr_labs)
test = (te_seqs, te_labs)
"""
f1 = open('C:/Users/7000030999/Downloads/sessionRec_NARM-master/train.pkl', 'wb')
pickle.dump(train, f1)
f1.close()
f2 = open('C:/Users/7000030999/Downloads/sessionRec_NARM-master/test.pkl', 'wb')
pickle.dump(test, f2)
f2.close()
"""
f1 = open('C:/Users/7000030999/Downloads/sessionRec_NARM-master/lastfm_preprocess/tsses_lastfmtrain_9month.txt', 'wb')
pickle.dump(train, f1)
f1.close()
f2 = open('C:/Users/7000030999/Downloads/sessionRec_NARM-master/lastfm_preprocess/tsses_lastfmtest_3month.txt', 'wb')
pickle.dump(test, f2)
f2.close()


print('Done.')