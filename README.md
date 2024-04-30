"CDS4SBRS: Content-driven Session for Session-Based Recommendation Systems"
This is our implementation for the paper:

## Datasets

1m-movielense dataset 
Goodreads-book dataset
lastfm dataset
Amazon dataset

## Notebooks and Helper Scripts
1.embedding_creation.py: This notebook contains a script that generates the embedding. We have linked our generated BERT description embeddings.
                          bert_embeddings_plot.txt - movielense  
                          bert_embeddings_books.txt- Goodreads-book
2.generate_clusters.py: This notebook used to generate cluster from description embeddings.

3.CD_session_creation.py : This notebook contains a script that generates content-driven sessions.

4.pipeline_genre.py: This notebook used to generate genre for track.

