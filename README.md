"Ready for You When You are Back: Content-driven Session-based Recommendation for Continuity of Experience"
This is our implementation for the paper:

## Requirements
python --> Use python 3.6.0 or newer.Python 2 is NOT supported.
numpy --> 1.15.1 or newer.
pandas --> 0.25.3 or newer.
CUDA --> Needed for the GPU support of Theano. It works fine with version 11.1.1.
libgpuarray --> Required for the GPU support of Theano, use the latest version.
theano --> 1.0.3  or newer. GPU support should be installed.

## Four Datasets used in this repo:-

1m-movielense dataset 
Goodreads-book dataset
lastfm dataset
Amazon dataset

## Content driven session creation using our proposed method

To replicate our CD-Sessions dataset, first download the dataset from the given link and place it into the respective dataset folder. Then, run the scripts below in the specified order.

For 1M-MovieLens dataset and Goodreads-books dataset, the following steps are involved:-
1.embedding_creation.py: This notebook contains a script that generates the embedding.
	Input file: descriptions.csv
	Output file: bert_embeddings.txt


2.generate_clusters.py: This notebook is used to generate clusters from description embeddings.
	Input file: bert_embeddings.txt and data.csv
	Output file:data_withclusterid.csv


3.CD_session_creation.py: This notebook contains a script that generates content-driven sessions.
	Input file: data_withclusterid.csv
	Output file:CDsess_data.csv

for lastfm_dataset, the following steps are involved.
1.pipeline_genre.py: This notebook used to generate genre for tracks.
	Inputfile:uniqueArtistandtrack.csv, after generating genre, we got the file unique_genres_with_clusterid.csv.
	outputfile:data_withclusterid.csv

2.CD_session_creation.py: This notebook contains a script that generates content-driven sessions.
	Input file: data_withclusterid.csv
	Output file:CDsess_data.csv

for Amazon_dataset, the following steps are involved.

1.CD_session_creation.py: This notebook contains a script that generates content-driven sessions.
	Input file: data_withclusterid.csv
	Output file:CDsess_data.csv

## Model Execution
We have leveraged our proposed Content-driven Sessions in the "Neural Attentive Recommendation Machine" model.

##Notebooks and Helper Scripts

1.example_preprocess.py: This notebook contains a script that can be used to create train and test data. we have used four example_preprocess.py (for example:ml_example_preprocess.py) for each dataset.

2.data_process.py: load train and test data to process data.

3.NARM.py: This notebook contains an implementation of a NARM model that leverages our proposed Content-driven Sessions.

Note: you can directly download train and test for each dataset from the respective dataset preprocess folder. here(link)


  
