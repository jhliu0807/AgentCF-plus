import pandas as pd
import sys
import os
import cornac
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..')))
from evaluation_cro_groupmem import get_similarity_score_list, max_retries, calculate_ndcg
from config import domain_list, random_domain0_source, random_domain1_source, random_domain2_source, random_domain3_source, item_data_source, get_main_kind, candidate_num
from dataPrepare import createRandomDF, createItemDF
from recommenders.utils.timer import Timer
from recommenders.models.cornac.cornac_utils import predict_ranking
SEED = 42

if __name__ == "__main__":
    inter_all_DF = pd.read_csv(f"dataset\crossDomainData\\user_item_data\{' '.join(domain_list)}\\timesequence\\inter_crossdomain_timesequence_all.csv", encoding="utf-8", dtype=str)
    inter_train_DF = pd.read_csv(f"dataset\crossDomainData\\user_item_data\{' '.join(domain_list)}\\timesequence\\inter_crossdomain_timesequence_train.csv", encoding="utf-8", dtype=str)
    inter_test_DF = pd.read_csv(f"dataset\crossDomainData\\user_item_data\{' '.join(domain_list)}\\timesequence\\inter_crossdomain_timesequence_test.csv", encoding="utf-8", dtype=str)
    # Construct a random selection dataset
    random_domain0_DF = createRandomDF(random_domain0_source)
    random_domain1_DF = createRandomDF(random_domain1_source)
    random_domain2_DF = createRandomDF(random_domain2_source)
    if len(domain_list) == 4:
        random_domain3_DF = createRandomDF(random_domain3_source)
    itemDF = createItemDF(item_data_source)

    # Create a mapping to convert string IDs to integer IDs
    user_mapping = {user: idx for idx, user in enumerate(inter_all_DF['user_id'].unique())}
    item_mapping = {item: idx for idx, item in enumerate(inter_all_DF['parent_asin'].unique())}
    # Replace original IDs with mapped IDs
    inter_train_DF['user_id'] = inter_train_DF['user_id'].map(user_mapping)
    inter_train_DF['parent_asin'] = inter_train_DF['parent_asin'].map(item_mapping)
    inter_train_DF = inter_train_DF[['user_id', 'parent_asin', 'rating']]
    train_set = cornac.data.Dataset.from_uir(inter_train_DF.itertuples(index=False), seed=SEED)

    # Train the BPR model
    bpr = cornac.models.BPR(
        k=200,
        max_iter=100,
        learning_rate=0.01,
        lambda_reg=0.001,
        verbose=True,
        seed=SEED
    )
    bpr.fit(train_set)

    all_predictions = predict_ranking(bpr, inter_train_DF, usercol="user_id", itemcol="parent_asin", remove_seen=True)
    # Evaluation
    ndcg_10_list = []
    ndcg_5_list = []
    ndcg_1_list = []
    mrr_list = []
    for index, record in inter_test_DF.iterrows():
        try:
            target_itemId = record["parent_asin"]
            userId = record["user_id"]
            main_kind = itemDF[itemDF["parent_asin"] == target_itemId]["main_category"].values[0]
            # Construct negative candidates
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

            # Randomly shuffle the data
            random.shuffle(random_itemId_list)
            random_itemId_list = [item_mapping[i] for i in random_itemId_list]
            target_itemId = item_mapping[target_itemId]
            userId = user_mapping[userId]
            item_score_list = []
            for i in random_itemId_list:
                result = all_predictions[(all_predictions['user_id'] == userId) & (all_predictions['parent_asin'] == i)]
                if len(result) == 0:
                    result = 0
                else:
                    result = all_predictions[(all_predictions['user_id'] == userId) & (all_predictions['parent_asin'] == i)]["prediction"].values[0]
                item_score_list.append(result)

            # Get the relevance score list
            relevance_score_list = [1 if x == target_itemId else 0 for x in random_itemId_list]
            target_rank = relevance_score_list.index(1) + 1  # Find the rank of the target
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
        file.write(f"\nNDCG@10:  {sum(ndcg_10_list)/len(ndcg_10_list)}\nNDCG@5:  {sum(ndcg_5_list)/len(ndcg_5_list)}\nNDCG@1:  {sum(ndcg_1_list)/len(ndcg_1_list)}\nMRR:  {sum(mrr_list)/len(mrr_list)}\n\n")