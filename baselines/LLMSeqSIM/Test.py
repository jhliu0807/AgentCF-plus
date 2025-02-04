import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..')))
from config import domain_list, item_data_source, random_domain0_source, random_domain1_source, random_domain2_source, random_domain3_source, get_main_kind, candidate_num
from evaluation import calculate_ndcg
from dataPrepare import createItemDF, createRandomDF
import random
import numpy as np
import pickle
from LLMSeqSIM import embedding_dims

seed = 42

def get_similarity_score_list(cdt_item_embedding, user_embedding):
    user_embedding = list(user_embedding)
    similarities = []
    # 遍历item dict，计算每个item embedding与user embedding的余弦相似度
    for item_id, item_embedding in cdt_item_embedding.items():
        cos_sim = np.dot(item_embedding, user_embedding) / (np.linalg.norm(item_embedding) * np.linalg.norm(user_embedding))
        # dot_prod_sim = np.dot(item_embedding, user_embedding)
        # euclidean_sim = -np.linalg.norm(np.array(item_embedding) - np.array(user_embedding))
        similarities.append((item_id, cos_sim))

    sorted_item_ids = [item_id for item_id, sim in sorted(similarities, key=lambda x: x[1], reverse=True)]
    
    return sorted_item_ids

if __name__ == "__main__":
    # 构造三张大表
    inter_test_DF = pd.read_csv(f"dataset\crossDomainData\\user_item_data\{' '.join(domain_list)}\\timesequence\\inter_crossdomain_timesequence_test.csv", encoding="utf-8", dtype=str)
    with open(f"baseline\\LLMSeqSIM\\item_embeddings_{' '.join(domain_list)}.pkl", 'rb') as f:
        item_embedding_dict = pickle.load(f)
    itemDF = createItemDF(item_data_source)

    # 构建随机选择数据集
    random_domain0_DF = createRandomDF(random_domain0_source)
    random_domain1_DF = createRandomDF(random_domain1_source)
    random_domain2_DF = createRandomDF(random_domain2_source)
    if len(domain_list) == 4:
        random_domain3_DF = createRandomDF(random_domain3_source)

    ndcg_10_list = []
    ndcg_5_list = []
    ndcg_1_list = []
    mrr_list = []
    for index, record in inter_test_DF.iterrows():
        try:
            target_itemId = record["parent_asin"]
            userId = record["user_id"]
            main_kind = itemDF[itemDF["parent_asin"] == target_itemId]["main_category"].values[0]
            # 构造neg candidate
            if main_kind == get_main_kind(domain_list[0]):
                random_itemId_list = [random_domain0_DF[random_domain0_DF["Unnamed: 0"] == userId][f"item_{random.randint(0, 99)}"].values[0] for i in range(candidate_num - 1)]
                random_itemId_list.append(target_itemId)
            elif main_kind == get_main_kind(domain_list[1]):
                random_itemId_list = [random_domain1_DF[random_domain1_DF["Unnamed: 0"] == userId][f"item_{random.randint(0, 99)}"].values[0] for i in range(candidate_num - 1)]
                random_itemId_list.append(target_itemId)
            elif main_kind == get_main_kind(domain_list[2]):
                random_itemId_list = [random_domain2_DF[random_domain2_DF["Unnamed: 0"] == userId][f"item_{random.randint(0, 99)}"].values[0] for i in range(candidate_num - 1)]
                random_itemId_list.append(target_itemId)
            elif len(domain_list) == 4 and main_kind == get_main_kind(domain_list[3]):
                random_itemId_list = [random_domain3_DF[random_domain3_DF["Unnamed: 0"] == userId][f"item_{random.randint(0, 99)}"].values[0] for i in range(candidate_num - 1)]
                random_itemId_list.append(target_itemId)

            # 随机打乱数据
            random.shuffle(random_itemId_list)
            cdt_item_embedding = {}
            for i in random_itemId_list:
                if i in item_embedding_dict.keys():
                    cdt_item_embedding[i] = item_embedding_dict[i]
                else:
                    cdt_item_embedding[i] = np.zeros(embedding_dims)

            # 读取user embedding
            with open(f"baseline\LLMSeqSIM\\user_embeddings_{' '.join(domain_list)}.pkl", "rb") as file:
                user_embedding_dict = pickle.load(file)
            if userId in user_embedding_dict.keys():
                user_embedding = user_embedding_dict[userId]
            else:
                user_embedding = np.zeros(embedding_dims)
            
            # 获取相似度列表
            similarity_score_list = get_similarity_score_list(cdt_item_embedding, user_embedding)
            relevance_score_list = [1 if x == target_itemId else 0 for x in similarity_score_list]
            target_rank = relevance_score_list.index(1) + 1  # 找到target的排名
            ndcg_at_10 = calculate_ndcg(relevance_score_list, 10)
            ndcg_10_list.append(ndcg_at_10)
            ndcg_at_5 = calculate_ndcg(relevance_score_list, 5)
            ndcg_5_list.append(ndcg_at_5)
            ndcg_at_1 = calculate_ndcg(relevance_score_list, 1)
            ndcg_1_list.append(ndcg_at_1)

            mrr_list.append(1.0/target_rank)
            print(f"ndcg@10:{ndcg_at_10} mean:{sum(ndcg_10_list) / len(ndcg_10_list)}")
            print(f"ndcg@5:{ndcg_at_5} mean:{sum(ndcg_5_list) / len(ndcg_5_list)}")
            print(f"ndcg@1:{ndcg_at_1} mean:{sum(ndcg_1_list) / len(ndcg_1_list)}")
            print(f"1/rank:{1.0/target_rank} mrr:{sum(mrr_list) / len(mrr_list)}")
        except Exception as e:
            print(e)
            continue
    with open(".\log\\result.txt", mode="a", encoding="utf-8") as file:
        file.write(f"\nSeed: {seed}\nNDCG@10:  {sum(ndcg_10_list)/len(ndcg_10_list)}\nNDCG@5:  {sum(ndcg_5_list)/len(ndcg_5_list)}\nNDCG@1:  {sum(ndcg_1_list)/len(ndcg_1_list)}\nMRR:  {sum(mrr_list)/len(mrr_list)}\n\n")
    
