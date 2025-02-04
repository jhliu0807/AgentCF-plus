import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import domain_list, item_data_source
from dataPrepare import createItemDF
from LLMSeqSIM import get_embeddings_batch, embedding_dims
import pickle

if __name__ == "__main__":
    # 1. Import the item interaction list for all users
    inter_all_df = pd.read_csv(f"dataset/crossDomainData/user_item_data/{' '.join(domain_list)}/timesequence/inter_crossdomain_timesequence_all.csv", encoding="utf-8", dtype=str)
    
    # Get unique item IDs
    item_id_list = list(set(inter_all_df["parent_asin"].tolist()))
    
    # Create item DataFrame and dictionary
    itemDF = createItemDF(item_data_source)
    item_dict = itemDF.set_index("parent_asin")["title"].to_dict()
    item_title_list = [item_dict.get(item_id) for item_id in item_id_list]

    # 3. Calculate embeddings for all items
    df = pd.DataFrame({'text': item_title_list, 'id': item_id_list})
    item_embeddings = {}

    for i in tqdm(range(len(df))):
        batch_texts = df['text'][i:i + 1].tolist()
        batch_embeddings = get_embeddings_batch(batch_texts, dim=embedding_dims, model='text-embedding-ada-002')
        item_embeddings[df["id"].iloc[i]] = batch_embeddings

    # Save item embeddings
    with open(f"baseline/LLMSeqSIM/item_embeddings_{' '.join(domain_list)}.pkl", 'wb') as f:
        pickle.dump(item_embeddings, f)

    exit()