import pandas as pd
import sys
import os
import random
from config import domain_list, item_data_source, get_main_kind, random_domain0_source, random_domain1_source, random_domain2_source, random_domain3_source, candidate_num, n_random_item
from dataPrepare import createItemDF, createRandomDF
from prompt import baseline_llmrank
from evaluation_cro_groupmem import calculate_ndcg

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    itemDF = createItemDF(item_data_source)

    # Start evaluation
    inter_test_DF = pd.read_csv(f"dataset/crossDomainData/user_item_data/{' '.join(domain_list)}/timesequence/inter_crossdomain_timesequence_test.csv", encoding="utf-8", dtype=str)

    # Build random selection datasets
    random_domains = [createRandomDF(source) for source in [random_domain0_source, random_domain1_source, random_domain2_source, random_domain3_source] if source]

    ndcg_10_list, ndcg_5_list, ndcg_1_list, mrr_list = [], [], [], []

    for index, record in inter_test_DF.iterrows():
        target_itemId = record["parent_asin"]
        userId = record["user_id"]
        main_kind = itemDF[itemDF["parent_asin"] == target_itemId]["main_category"].values[0]

        # Construct negative candidates
        for i, domain in enumerate(random_domains):
            if main_kind == get_main_kind(domain_list[i]):
                random_itemId_list = [domain[domain["Unnamed: 0"] == userId][f"item_{random.randint(0, n_random_item - 1)}"].values[0] for _ in range(candidate_num - 1)]
                random_itemId_list.append(target_itemId)
                break

        # Shuffle the data
        random.shuffle(random_itemId_list)

        # Get candidate popularity
        cdt_item_popularity_list = []
        for cdt_itemId in random_itemId_list:
            try:
                cdt_item_popularity_list.append(itemDF[itemDF["parent_asin"] == cdt_itemId]["rating_number"].values[0])
            except Exception as e:
                print(f"Error processing item {cdt_itemId}: {e}")
                cdt_item_popularity_list.append(0)

        # Sort candidates by popularity
        sorted_items = sorted(zip(random_itemId_list, cdt_item_popularity_list), key=lambda x: x[1], reverse=True)
        sorted_item_id_list = [item[0] for item in sorted_items]
        relevance_score_list = [1 if x == target_itemId else 0 for x in sorted_item_id_list]

        # Calculate NDCG and MRR
        try:
            target_rank = relevance_score_list.index(1) + 1  # Find the target rank
            ndcg_10_list.append(calculate_ndcg(relevance_score_list, 10))
            ndcg_5_list.append(calculate_ndcg(relevance_score_list, 5))
            ndcg_1_list.append(calculate_ndcg(relevance_score_list, 1))
            mrr_list.append(1.0 / target_rank)

            print(f"ndcg@10: {ndcg_10_list[-1]} mean: {sum(ndcg_10_list) / len(ndcg_10_list)}")
            print(f"ndcg@5: {ndcg_5_list[-1]} mean: {sum(ndcg_5_list) / len(ndcg_5_list)}")
            print(f"ndcg@1: {ndcg_1_list[-1]} mean: {sum(ndcg_1_list) / len(ndcg_1_list)}")
            print(f"1/rank: {1.0 / target_rank} mrr: {sum(mrr_list) / len(mrr_list)}")
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            continue

    # Save results to a log file
    with open("./log/result.txt", mode="a", encoding="utf-8") as file:
        file.write(f"NDCG@10: {sum(ndcg_10_list) / len(ndcg_10_list)}\n"
                   f"NDCG@5: {sum(ndcg_5_list) / len(ndcg_5_list)}\n"
                   f"NDCG@1: {sum(ndcg_1_list) / len(ndcg_1_list)}\n"
                   f"MRR: {sum(mrr_list) / len(mrr_list)}\n\n")