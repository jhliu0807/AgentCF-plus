import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from config import domain_list, item_data_source
from dataPrepare import createItemDF
from openai import OpenAI

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = ''
client = OpenAI()

embedding_dims = 8

def normalize_l2(x):
    x = np.array(x)
    norm = np.linalg.norm(x, axis=1, keepdims=True) if x.ndim > 1 else np.linalg.norm(x)
    return np.where(norm == 0, x, x / norm)

def get_embeddings_batch(texts, dim, model="text-embedding-ada-002"):
    texts = [text.replace("\n", " ") for text in texts]
    response = client.embeddings.create(input=texts, model=model)
    return [response.data[i].embedding[:dim] for i in range(len(response.data))]

if __name__ == "__main__":
    # 1. Import item interaction list for each user
    inter_train_df = pd.read_csv(f"dataset/crossDomainData/user_item_data/{' '.join(domain_list)}/timesequence/inter_crossdomain_timesequence_train.csv", encoding="utf-8", dtype=str)
    
    # Load item embeddings
    with open(f"baseline/LLMSeqSIM/item_embeddings_{' '.join(domain_list)}.pkl", 'rb') as f:
        item_embedding_dict = pickle.load(f)

    # Group user interactions
    combined_user_item_df = inter_train_df.groupby('user_id')['parent_asin'].apply(list).reset_index()
    combined_user_item_df.columns = ['user_id', 'parent_asin_list']
    combined_user_item_df.set_index('user_id', inplace=True)

    # 2. Write the names of items interacted with by each user
    itemDF = createItemDF(item_data_source)
    item_dict = itemDF.set_index("parent_asin")["title"].to_dict()
    combined_user_item_df['item_title_list'] = combined_user_item_df['parent_asin_list'].apply(lambda ids: [item_dict[item_id] for item_id in ids])

    # 3. Calculate user embeddings
    user_embeddings = {}
    for index, row in combined_user_item_df.iterrows():
        embeddings = [item_embedding_dict[i] for i in row["parent_asin_list"]]
        user_embeddings[row.name] = embeddings[-1]  # Use the last embedding

    # Save user embeddings
    with open(f"user_group_mem/user_group_via_history/user_embeddings_{' '.join(domain_list)}.pkl", 'wb') as f:
        pickle.dump(user_embeddings, f)

    exit()