import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm  

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(description):
    inputs = tokenizer(description, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()
    return cls_embedding

data = pd.read_csv('matched_bookId_descriptions.csv',encoding='utf-8-sig')

with open('bert_embeddings_books.txt', 'w') as out_file:
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        item_id = row['book_id']
        description = row['Description']
        embedding = get_bert_embedding(description)
        embedding_str = ' '.join(map(str, embedding))
        out_file.write(f"{item_id} {embedding_str}\n")
