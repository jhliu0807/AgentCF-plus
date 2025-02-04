import sys
import os
import pandas as pd
import numpy as np
import random
import pickle
from sasrec.model import SASREC
from sasrec.sampler import WarpSampler
from sasrec.util import SASRecDataSet
from config import domain_list, item_data_source, random_domain0_source, random_domain1_source, random_domain2_source, random_domain3_source, get_main_kind, candidate_num
from dataPrepare import createInterDF, createItemDF, createRandomDF
from evaluation_cro_groupmem import calculate_ndcg

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..')))

if __name__ == "__main__":
    # Load interaction data
    inter_all_DF = pd.read_csv(f"dataset/crossDomainData/user_item_data/{' '.join(domain_list)}/timesequence/inter_crossdomain_timesequence_all.csv", encoding="utf-8", dtype=str)
    inter_all_DF = (inter_all_DF.rename(columns={'user_id': 'userID', 'parent_asin': 'itemID', 'timestamp': 'time'})
                    .sort_values(by=['userID', 'time'])
                    .drop(['rating', 'time'], axis=1)
                    .reset_index(drop=True)[["userID", "itemID"]])

    user_set, item_set = set(inter_all_DF['userID'].unique()), set(inter_all_DF['itemID'].unique())
    user_map = {user: idx + 1 for idx, user in enumerate(user_set)}
    item_map = {item: idx + 1 for idx, item in enumerate(item_set)}

    inter_all_DF["userID"] = inter_all_DF["userID"].map(user_map)
    inter_all_DF["itemID"] = inter_all_DF["itemID"].map(item_map)

    # Save SASRec data
    inter_all_DF.to_csv('baseline/SASRec/sasrec_data.txt', sep="\t", header=False, index=False)
    
    # Save maps
    with open('baseline/SASRec/maps.pkl', 'wb') as f:
        pickle.dump((user_map, item_map), f)

    # Prepare the dataset
    data = SASRecDataSet('baseline/SASRec/sasrec_data.txt')
    data.split()  # Train, validation, test split

    # Model parameters
    max_len = 80
    hidden_units = 8
    batch_size = 128

    model = SASREC(
        item_num=data.itemnum,
        seq_max_len=max_len,
        num_blocks=1,
        embedding_dim=hidden_units,
        attention_dim=hidden_units,
        attention_num_heads=1,
        dropout_rate=0.4,
        conv_dims=[hidden_units, hidden_units],
        l2_reg=0.00001
    )

    sampler = WarpSampler(data.user_train, data.usernum, data.itemnum, batch_size=batch_size, maxlen=max_len, n_workers=1)
    model.build((None, max_len))
    model.train(
        data,
        sampler,
        num_epochs=3,
        batch_size=batch_size,
        lr=0.001,
        val_epoch=1,
        val_target_user_n=100,
        target_item_n=-1,
        auto_save=True,
        path="baseline/SASRec",
        exp_name='exp_example',
    )

    # Sample target users
    model.sample_val_users(data, 100)
    encoded_users = model.val_users

    # Load test data
    inter_test_DF = pd.read_csv(f"dataset/crossDomainData/user_item_data/{' '.join(domain_list)}/timesequence/inter_crossdomain_timesequence_test.csv", encoding="utf-8", dtype=str)
    itemDF = createItemDF(item_data_source)

    # Create random datasets
    random_domains = [createRandomDF(source) for source in [random_domain0_source, random_domain1_source, random_domain2_source, random_domain3_source] if source]
    
    ndcg_10_list, ndcg_5_list, ndcg_1_list, mrr_list = [], [], [], []

    for index, record in inter_test_DF.iterrows():
        try:
            target_itemId = record["parent_asin"]
            userId = record["user_id"]
            main_kind = itemDF[itemDF["parent_asin"] == target_itemId]["main_category"].values[0]

            # Construct negative candidates
            random_itemId_list = []
            for i, domain in enumerate(random_domains):
                if main_kind == get_main_kind(domain_list[i]):
                    random_itemId_list = [domain[domain["Unnamed: 0"] == userId][f"item_{random.randint(0, 99)}"].values[0] for _ in range(candidate_num - 1)]
                    random_itemId_list.append(target_itemId)
                    break

            random.shuffle(random_itemId_list)

            # Get scores
            score = model.get_user_item_score(data, [userId], random_itemId_list, user_map, item_map, batch_size=1)
            score = score.transpose().iloc[1:]
            sorted_df = score.sort_values(by=0, ascending=False)
            index_list = sorted_df.index.tolist()
            relevance_score_list = [1 if x == target_itemId else 0 for x in index_list]

            target_rank = relevance_score_list.index(1) + 1
            ndcg_10_list.append(calculate_ndcg(relevance_score_list, 10))
            ndcg_5_list.append(calculate_ndcg(relevance_score_list, 5))
            ndcg_1_list.append(calculate_ndcg(relevance_score_list, 1))
            mrr_list.append(1.0 / target_rank)

            print(f"ndcg@10: {ndcg_10_list[-1]} mean: {sum(ndcg_10_list) / len(ndcg_10_list)}")
            print(f"ndcg@5: {ndcg_5_list[-1]} mean: {sum(ndcg_5_list) / len(ndcg_5_list)}")
            print(f"ndcg@1: {ndcg_1_list[-1]} mean: {sum(ndcg_1_list) / len(ndcg_1_list)}")
            print(f"1/rank: {1.0 / target_rank} mrr: {sum(mrr_list) / len(mrr_list)}")
        except Exception as e:
            print(e)
            continue

    with open(".\log\result.txt", mode="a", encoding="utf-8") as file:
        file.write(f"\nNDCG@10: {sum(ndcg_10_list) / len(ndcg_10_list)}\nNDCG@5: {sum(ndcg_5_list) / len(ndcg_5_list)}\nNDCG@1: {sum(ndcg_1_list) / len(ndcg_1_list)}\nMRR: {sum(mrr_list) / len(mrr_list)}\n\n")

    exit()