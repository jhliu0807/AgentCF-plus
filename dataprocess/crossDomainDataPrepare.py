import random
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os
from collections import Counter
from config import domain_list, n_random_item
from dataPrepare import prepare_initial_mem_from_interDF, prepare_data_from_interDF

random_seed = 23
random.seed(random_seed)

def select_100_user_data(domain_list):
    inter_dfs = []
    for domain in domain_list:
        df = pd.read_csv(f"dataset/crossDomainData/filtered_data/inter/inter_{domain}.csv", encoding="utf-8")
        df = df[df['rating'] >= 4]
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[(df['datetime'] >= datetime(2021, 10, 1)) & (df['datetime'] <= datetime(2022, 3, 31))]
        inter_dfs.append(df)

    # Calculate user interactions for each domain
    user_interactions = [df.groupby("user_id").size().reset_index(name='interactions') for df in inter_dfs]

    # Keep only common users
    all_userids = pd.concat([ui['user_id'] for ui in user_interactions])
    common_userids = set(all_userids.value_counts()[all_userids.value_counts() >= 2].index)

    for i in range(len(user_interactions)):
        user_interactions[i] = user_interactions[i][user_interactions[i]["user_id"].isin(common_userids)]
        user_interactions[i] = user_interactions[i].rename(columns={"interactions": f"interactions_domain{i}"})

    # Merge all user interactions
    user_interactions_all = user_interactions[0]
    for i in range(1, len(user_interactions)):
        user_interactions_all = pd.merge(user_interactions_all, user_interactions[i], on="user_id", how="outer").fillna(0)

    # Filter users based on interaction counts
    user_interactions_all["sum_interactions"] = user_interactions_all.iloc[:, 1:].sum(axis=1)
    user_interactions_all = user_interactions_all[(user_interactions_all["sum_interactions"] >= 10) & (user_interactions_all["sum_interactions"] <= 100)]

    # Randomly select 100 users
    selected_user_set = set(user_interactions_all["user_id"].sample(n=100, random_state=random_seed).tolist())

    # Create a dictionary to store each user's interacted items
    user_domain_dicts = {}
    for i, df in enumerate(inter_dfs):
        user_items = {}
        for user_id, group in df[df["user_id"].isin(selected_user_set)].groupby("user_id"):
            user_items[user_id] = list(set(group["parent_asin"].astype(str)))
        user_domain_dicts[f"user_domain{i}"] = pd.DataFrame.from_dict(user_items, orient='index', columns=[f'col_{j}' for j in range(max(len(v) for v in user_items.values()))])
        user_domain_dicts[f"user_domain{i}"] = user_domain_dicts[f"user_domain{i}"].astype(str)

        # Count items
        item_counts = Counter(user_domain_dicts[f"user_domain{i}"].stack())
        print(len(item_counts))

    output_dir = f"dataset/crossDomainData/user_item_data/{' '.join(domain_list)}/inter"
    os.makedirs(output_dir, exist_ok=True)

    for i, domain in enumerate(domain_list):
        user_domain_dicts[f"user_domain{i}"].to_csv(f"{output_dir}/inter_user_{domain}.csv")

def select_random_item(domain_list):
    selected_inter_dfs = []
    all_inter_dfs = []
    all_items = []

    for domain in domain_list:
        selected_inter_df = pd.read_csv(f"dataset/crossDomainData/user_item_data/{' '.join(domain_list)}/inter/inter_user_{domain}.csv", dtype=str, encoding="utf-8")
        selected_inter_dfs.append(selected_inter_df)

        all_inter_df = pd.read_csv(f"dataset/crossDomainData/filtered_data/inter/inter_{domain}.csv", encoding="utf-8")
        all_inter_df = all_inter_df[all_inter_df["rating"] >= 4]
        all_inter_dfs.append(all_inter_df)

        all_items.append(set(selected_inter_df.iloc[:, 1:].stack().astype(str)))

    sampled_user_set = set()
    for selected_inter_df in selected_inter_dfs:
        sampled_user_set.update(set(selected_inter_df["Unnamed: 0"]))

    for i, all_inter_df in enumerate(all_inter_dfs):
        all_inter_dfs[i] = all_inter_df[all_inter_df["user_id"].isin(sampled_user_set)]

    user_to_items = [all_inter_df.groupby('user_id')['parent_asin'].apply(set).to_dict() for all_inter_df in all_inter_dfs]

    random_items_dfs = [pd.DataFrame(0, index=list(sampled_user_set), columns=range(n_random_item)) for _ in domain_list]

    for user in sampled_user_set:
        for i, user_items in enumerate(user_to_items):
            if user in user_items:
                interacted_items = user_items[user]
                non_interacted_items = list(all_items[i] - interacted_items)
                random_items = random.sample(non_interacted_items, n_random_item)
                random_items_dfs[i].loc[user, :] = random_items

    for i in range(len(random_items_dfs)):
        random_items_dfs[i].columns = [f'item_{j}' for j in range(n_random_item)]

    output_dir = f"dataset/crossDomainData/user_item_data/{' '.join(domain_list)}/random"
    os.makedirs(output_dir, exist_ok=True)

    for i, domain in enumerate(domain_list):
        random_items_dfs[i].to_csv(f"{output_dir}/random_{domain}.csv", encoding="utf-8")

def process(domain_list): 
    inter_user_dfs = [pd.read_csv(f"dataset/crossDomainData/user_item_data/{' '.join(domain_list)}/inter/inter_user_{domain}.csv", encoding="utf-8") for domain in domain_list]

    sampled_user_set = set()
    for df in inter_user_dfs:
        sampled_user_set.update(set(df["Unnamed: 0"]))

    all_inter_dfs = [pd.read_csv(f"dataset/crossDomainData/filtered_data/inter/inter_{domain}.csv", encoding="utf-8") for domain in domain_list]

    for i in range(len(all_inter_dfs)):
        all_inter_dfs[i] = all_inter_dfs[i][all_inter_dfs[i]["user_id"].isin(sampled_user_set)]

    selected_inter_dfs = [[] for _ in domain_list]

    for i, df in enumerate(inter_user_dfs):
        for index, row in df.iterrows():
            userid = row['Unnamed: 0']
            for parentasin in row.iloc[1:]:
                parentasin = str(parentasin)
                if parentasin == "nan":
                    break
                matching_rows = all_inter_dfs[i][(all_inter_dfs[i]['user_id'] == userid) & (all_inter_dfs[i]['parent_asin'] == parentasin)]
                selected_inter_dfs[i].append(matching_rows)

    selected_inter_dfs = [pd.concat(selected_inter_df, ignore_index=True) for selected_inter_df in selected_inter_dfs]

    for i in range(len(selected_inter_dfs)):
        selected_inter_dfs[i] = selected_inter_dfs[i].sort_values(by='timestamp')

    output_dir = f"dataset/crossDomainData/user_item_data/{' '.join(domain_list)}/timesequence"
    os.makedirs(output_dir, exist_ok=True)

    for i, domain in enumerate(domain_list):
        selected_inter_dfs[i].to_csv(f"{output_dir}/inter_{domain}_timesequence_all.csv", encoding="utf-8")

def mergeCrossDomainData(domain_list):
    item_dfs = [pd.read_csv(f"dataset/crossDomainData/filtered_data/meta/meta_{domain}.csv") for domain in domain_list]
    item_all_df = pd.concat(item_dfs, ignore_index=True)
    item_all_df.to_csv(f"dataset/crossDomainData/user_item_data/{' '.join(domain_list)}/meta_crossdomain.csv", index=False)

    user_inter_dfs = [pd.read_csv(f"dataset/crossDomainData/user_item_data/{' '.join(domain_list)}/timesequence/inter_{domain}_timesequence_all.csv") for domain in domain_list]
    user_inter_timesequence_all = pd.concat(user_inter_dfs, ignore_index=True).sort_values(by='timestamp')

    num_all_data = len(user_inter_timesequence_all)
    num_train_data = int(num_all_data * 0.9)

    user_inter_timesequence_all.to_csv(f"dataset/crossDomainData/user_item_data/{' '.join(domain_list)}/timesequence/inter_crossdomain_timesequence_all.csv", index=False)
    user_inter_timesequence_all.iloc[:num_train_data].to_csv(f"dataset/crossDomainData/user_item_data/{' '.join(domain_list)}/timesequence/inter_crossdomain_timesequence_train.csv", index=False)
    user_inter_timesequence_all.iloc[num_train_data:].to_csv(f"dataset/crossDomainData/user_item_data/{' '.join(domain_list)}/timesequence/inter_crossdomain_timesequence_test.csv", index=False)

if __name__ == "__main__":
    # Uncomment the following lines as needed
    # filter_meta_data()  
    # filter_inter_data()  
    # select_100_user_data(domain_list)
    select_random_item(domain_list)
    # process(domain_list)
    # mergeCrossDomainData(domain_list)
    # prepare_initial_mem_from_interDF('all', domain_list)
    # prepare_data_from_interDF('all', domain_list, crossDomain=cross_domain)
    
    exit()