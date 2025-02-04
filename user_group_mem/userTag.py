'''
This script uses the user's cross-domain memory to tag each user with interest labels.
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import model, domain_list, group_n_cluster
import pandas as pd
from prompt import get_user_tag_prompt
from request import get_response_from_openai
import json
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from openai import OpenAI
from functions import concatenate_crossdomain_preference

class Args:
    def __init__(self, domain_list):
        self.exp_name = "AgentCF++" + " " + " ".join(domain_list)
        self.model = 'text-embedding-3-large'
        self.dim = 128
        self.batch_size = 1
    @property
    def name_suffix(self):
        return self.exp_name.replace("AgentCF++ ", "")
    @property
    def dataset_name(self):
        return 'user_tag' + " " + self.name_suffix
    @property
    def input_file(self):
        return f'user_group_mem/llm4embedding/input/{self.dataset_name}.json'
    @property
    def output_file(self):
        return f'user_group_mem/llm4embedding/output/{self.dataset_name}_embedding.npy'

os.environ['OPENAI_API_KEY'] = ''
client = OpenAI()

def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)

def get_embeddings_batch(texts, dim=64, model="text-embedding-3-small"):
    texts = [text.replace("\n", " ") for text in texts]
    response = client.embeddings.create(input=texts, model=model)
    return [normalize_l2(response.data[i].embedding[:dim]) for i in range(len(response.data))]

def gen_user_tag_dict(user_id_all, exp_name, name_suffix):
    user_tag_dict = {}
    for userId in user_id_all:
        private_domain_description = concatenate_crossdomain_preference(f"memory/{exp_name}/user/user.{userId}")
        user_tag_prompt = get_user_tag_prompt(private_domain_description)
        responseText = get_response_from_openai(user_tag_prompt, model)
        data = json.loads(responseText)
        user_tag_dict[userId] = data["interest_tags"]

    save_path = f"user_group_mem/llm4embedding/input/user_tag {name_suffix}.json"
    with open(save_path, 'w') as file:
        json.dump(user_tag_dict, file)
    return user_tag_dict

def process(args, ratio):
    # 1. Get all user IDs in the training set
    inter_timesequence_df = pd.read_csv(f"dataset/crossDomainData/user_item_data/{' '.join(domain_list)}/timesequence/inter_crossdomain_timesequence_train.csv", encoding="utf-8")
    inter_timesequence_df = inter_timesequence_df[["user_id", "parent_asin"]]
    user_id_all = inter_timesequence_df["user_id"].unique()

    # 2. For each user, use the current user memory to get labels from the large model, resulting in a user-tag matrix.
    user_tag_dict = gen_user_tag_dict(user_id_all, args.exp_name, args.name_suffix)

    # 3. Generate embeddings using the large model
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_tags = [tag for tags in data.values() for tag in tags]
    df = pd.DataFrame(all_tags, columns=['text'])

    with open(f"user_group_mem/llm4embedding/output/column_list {args.name_suffix}.json", "w", encoding="utf-8") as file:
        json.dump(list(df['text']), file)
    embeddings = []
    for i in tqdm(range(0, len(df), args.batch_size)):
        batch_texts = df['text'][i:i + args.batch_size].tolist()
        batch_embeddings = get_embeddings_batch(batch_texts, dim=args.dim, model=args.model)
        embeddings.extend(batch_embeddings)
    np.save(args.output_file, embeddings)

    # Perform k-means clustering on the embeddings
    embedding_array = np.load(args.output_file)
    n_clusters = int(group_n_cluster * ratio / 10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embedding_array)
    labels = kmeans.labels_

    # Combine clustering results with the text of the labels
    with open(f"user_group_mem/llm4embedding/output/column_list {args.name_suffix}.json", encoding="utf-8") as file:
        tags = json.load(file)
    tags_cluster_df = pd.DataFrame(labels, index=tags, columns=["cluster"])
    cluster_tags_df = tags_cluster_df.groupby("cluster").apply(lambda x: x.index.tolist()).reset_index()
    cluster_tags_df.to_csv(f"user_group_mem/output/cluster_tags {args.name_suffix}.csv", index=False)

if __name__ == "__main__":
    args = Args(domain_list)
    process(args, 10)