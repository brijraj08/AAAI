# STAMP
---
We have leveraged our proposed Content-driven Sessions in the "STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation"

These are four datasets we used in our paper. After download them, you can put them in the folder `datas\`, then process them by `process_ml.py`,  `process_book.py`,`process_lastfm.py`,`process_Amazon.py` respectively.
---

## Usage

run the file`ml_cmain.py`,`book_cmain.py`, `lastfm_cmain.py` ,`Amazon_cmain.py` for each dataset to train the model.

For example: python ml_cmain.py -m stamp_rsc -d rsc15_64 -n -r

---
## Requirements

. Python 3
. Tensorflow 1.4


