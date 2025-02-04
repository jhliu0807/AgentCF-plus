import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..')))
from config import domain_list, item_data_source, get_main_kind, random_domain0_source, random_domain1_source, random_domain2_source, random_domain3_source, candidate_num, evaluation_times, model
from dataPrepare import createItemDF, createRandomDF
import random
from prompt import baseline_llmrank
from evaluation_cro_groupmem import get_similarity_score_list, max_retries, calculate_ndcg

if __name__ == "__main__":
    ### 1. Import the item interaction list for each user
    inter_train_df = pd.read_csv(f"dataset\crossDomainData\\user_item_data\{' '.join(domain_list)}\\timesequence\\inter_crossdomain_timesequence_train.csv", encoding="utf-8", dtype=str)
    user_item_df = inter_train_df.groupby('user_id')['parent_asin'].apply(list).reset_index()
    # Rename columns
    user_item_df.columns = ['user_id', 'parent_asin_list']
    # Set user_id as the index
    user_item_df.set_index('user_id', inplace=True)
    ### 2. Write the names of items interacted with by each user
    itemDF = createItemDF(item_data_source)
    item_dict = itemDF[["parent_asin", "title"]].set_index("parent_asin")["title"].to_dict()
    user_item_df['item_title_list'] = user_item_df['parent_asin_list'].apply(lambda ids: [item_dict[item_id] for item_id in ids])

    ## Start evaluation
    inter_test_DF = pd.read_csv(f"dataset\crossDomainData\\user_item_data\{' '.join(domain_list)}\\timesequence\\inter_crossdomain_timesequence_test.csv", encoding="utf-8", dtype=str)
    # Construct a random selection dataset
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
        target_itemId = record["parent_asin"]
        userId = record["user_id"]
        target_item_title = str(itemDF[itemDF["parent_asin"] == target_itemId]["title"].values[0])
        if len(user_item_df[user_item_df.index == userId]["item_title_list"].values) == 0:
            user_his_text = "Temporarily unavailable"
        else:
            user_his_text_list = user_item_df[user_item_df.index == userId]["item_title_list"].values[0]
            user_his_text = ""
            for title in user_his_text_list:
                user_his_text += "title:"+str(itemDF[itemDF["title"] == title]["title"].values[0])+ "|| subtitle:"+str(itemDF[itemDF["title"] == title]["subtitle"].values[0]) + "||main_category:"+str(itemDF[itemDF["title"] == title]["main_category"].values[0]) + "|| average_rating:"+str(itemDF[itemDF["title"] == title]["average_rating"].values[0]) + "|| rating_number:"+str(itemDF[itemDF["title"] == title]["rating_number"].values[0]) + "|| price:"+str(itemDF[itemDF["title"] == title]["price"].values[0]) + "|| store:"+str(itemDF[itemDF["title"] == title]["store"].values[0]) + "|| item id:"+str(itemDF[itemDF["title"] == title]["parent_asin"].values[0])

        main_kind = itemDF[itemDF["parent_asin"] == target_itemId]["main_category"].values[0]
        # Construct negative candidates
        if main_kind == get_main_kind(domain_list[0]):
            main_kind = domain_list[0]
            random_itemId_list = [random_domain0_DF[random_domain0_DF["Unnamed: 0"] == userId][f"item_{random.randint(0, 99)}"].values[0] for i in range(candidate_num - 1)]
            random_itemId_list.append(target_itemId)
        elif main_kind == get_main_kind(domain_list[1]):
            main_kind = domain_list[1]
            random_itemId_list = [random_domain1_DF[random_domain1_DF["Unnamed: 0"] == userId][f"item_{random.randint(0, 99)}"].values[0] for i in range(candidate_num - 1)]
            random_itemId_list.append(target_itemId)
        elif main_kind in get_main_kind(domain_list[2]):
            main_kind = domain_list[2]
            random_itemId_list = [random_domain2_DF[random_domain2_DF["Unnamed: 0"] == userId][f"item_{random.randint(0, 99)}"].values[0] for i in range(candidate_num - 1)]
            random_itemId_list.append(target_itemId)
        elif len(domain_list) == 4 and main_kind in get_main_kind(domain_list[3]):
            main_kind = domain_list[3]
            random_itemId_list = [random_domain3_DF[random_domain3_DF["Unnamed: 0"] == userId][f"item_{random.randint(0, 99)}"].values[0] for i in range(candidate_num - 1)]
            random_itemId_list.append(target_itemId)

        # Randomly shuffle the data
        random.shuffle(random_itemId_list)

        # Read candidate's related information
        cdt_item_title_list = []
        for cdt_itemId in random_itemId_list:
            try:
                cdt_item_title_list.append("title:"+str(itemDF[itemDF["parent_asin"] == cdt_itemId]["title"].values[0]) + "||main_category:"+str(itemDF[itemDF["parent_asin"] == cdt_itemId]["main_category"].values[0]) + "||subtitle:"+str(itemDF[itemDF["parent_asin"] == cdt_itemId]["subtitle"].values[0]) + "|| price:"+str(itemDF[itemDF["parent_asin"] == cdt_itemId]["price"].values[0]) + "|| average_rating:"+str(itemDF[itemDF["parent_asin"] == cdt_itemId]["average_rating"].values[0]) + "|| rating_number:"+str(itemDF[itemDF["parent_asin"] == cdt_itemId]["rating_number"].values[0]) + "|| item id:"+str(itemDF[itemDF["parent_asin"] == cdt_itemId]["parent_asin"].values[0]))

            except Exception as e:
                # If any exception occurs, print the error message and continue the loop
                print(f"Error processing item {cdt_itemId}: {e}")
                cdt_item_title_list.append('nan')
                continue
        # Construct item prompt
        list_of_item_title = ''
        for cdt_item_title in cdt_item_title_list:
            list_of_item_title += f"{cdt_item_title.strip()};\n"

        # Create prompt
        system_evaluation_prompt = baseline_llmrank(user_his_text=user_his_text, recent_item=user_his_text[-1], recall_budget=candidate_num, candidate_text_order="\n".join(cdt_item_title_list))

        # Sort multiple times to reduce randomness
        for i in range(evaluation_times):
            # Get the similarity score list
            similarity_score_list = get_similarity_score_list(system_evaluation_prompt, model, target_item_title)
            retries = 0
            # If there is a problem with the similarity score list, regenerate
            while len(similarity_score_list) != candidate_num and retries < max_retries:
                retries += 1
                print(f"retry {retries} ...")
                similarity_score_list = get_similarity_score_list(system_evaluation_prompt, model, target_item_title)

            relevance_score_list = [1 if x == max(similarity_score_list) else 0 for x in similarity_score_list]
            try:
                target_rank = relevance_score_list.index(1) + 1  # Find the rank of the target
            except Exception as e:
                print(f"{e}")
                continue
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

    with open(".\log\\result.txt", mode="a", encoding="utf-8") as file:
        file.write(f"NDCG@10:  {sum(ndcg_10_list)/len(ndcg_10_list)}\nNDCG@5:  {sum(ndcg_5_list)/len(ndcg_5_list)}\nNDCG@1:  {sum(ndcg_1_list)/len(ndcg_1_list)}\nMRR:  {sum(mrr_list)/len(mrr_list)}\n\n")
    exit()