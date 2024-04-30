#!pip install -U -q google.generativeai
import pandas as pd
import re

import numpy as np
import pandas as pd
import google.generativeai as genai
import google.ai.generativelanguage as glm

from IPython.display import Markdown
#put api key here
genai.configure(api_key="")

def get_genres_batch(artists, tracks):
        prompts = ["Generate the genre for the following songs and output only the genre :\n" + "\n".join([f"{i}) {track} by {artist}" for i, (artist, track) in enumerate(zip(artists, tracks), start=1)])]
        model = genai.GenerativeModel('gemini-pro')
        answers = model.generate_content(prompts)

        return [answer.text for answer in answers]


df=pd.read_csv(r'uniqueArtistandtrack.csv',encoding='utf-8-sig')
batch_size = 100
total_entries = df.shape[0]

def check_size(genres_list, expected_size):
    return len(genres_list) == expected_size or (len(genres_list) == 1 and expected_size != 1)

for start_idx in range(0, total_entries, batch_size):
    attempts = 0
    success = False

    while attempts < 5 and not success:
        try:
            end_idx = start_idx + batch_size
            batch_df = df.iloc[start_idx:end_idx]

            artists = batch_df['artist_name'].tolist()
            tracks = batch_df['track_name'].tolist()

            genre_response = get_genres_batch(artists, tracks)
            genres_str = str(genre_response[0])
            genres_str_clean = x=re.sub('[\[ \] 0-9 \)]',"",genres_str)
            genres_list = genres_str_clean.split('\n')
            if not check_size(genres_list, len(batch_df)):
                raise ValueError("Size error")
            print(genres_list)
            for i, genre in enumerate(genres_list):
                if i < len(batch_df):
                    df.at[start_idx + i, 'genre'] = genre

            success = True
            print(f"added genres for batch starting at {start_idx}")
        except ValueError as e:
            attempts += 1
            print(f"{attempts} failed for batch at index {start_idx}: {e}")

    if not success:
        print(f"Failed to add genres for batch starting at index {start_idx} ")

    if (start_idx // batch_size) % 1 == 0:
        df.to_csv('output_file_partial.csv', index=False)
        print(f"Progress saved to 'output_file_partial.csv' after processing up to entry {end_idx}")
df.to_csv('output_file_final.csv', index=False)
print("Final data saved to 'output_file_final.csv'")