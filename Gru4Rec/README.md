# GRU4Rec
We have leveraged our proposed Content-driven Sessions in the "Session-based Recommendations With Recurrent Neural Networks"(https://arxiv.org/abs/1511.06939 "Session-based Recommendations With Recurrent Neural Networks")

## Requirements

- **python** --> Use python `3.6.3` or newer. 
- **numpy** --> `1.16.4` or newer.
- **pandas** --> `0.24.2` or newer.
- **CUDA** --> Needed for the GPU support of Theano.It works fine with more recent versions, e.g. `11.8`.
- **libgpuarray** --> Required for the GPU support of Theano, use the latest version.
- **theano** --> `1.0.5` (last stable release) or newer (occassionally it is still updated with minor stuff). GPU support should be installed.

## Usage

### Execute experiments
Train, save and evaluate a model measuring recall and MRR at 1, 5, 10 and 20 using model parameters from a parameter string.
for ex for  bookdataset:
$ THEANO_FLAGS=device=cuda0 python run.py /GRU4Rec-master/examples/data/ -t /GRU4Rec-master/examples/data/ -m 10 20 -ps layers=100,batch_size=80,dropout_p_embed=0.5,dropout_p_hidden=0.05,learning_rate=0.05,momentum=0.4,n_sample=2048,sample_alpha=0.4,bpreg=1.95,logq=0.0,loss=bpr-max,constrained_embedding=True,final_act=elu-0.5,n_epochs=10 -s /path/to/Gru4rec_save_model.pickle


