"""
This script is used to build group memory
"""
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import domain_list, get_main_kind

def process(exp_name, name_suffix, ratio):
    # Import group-user table
    group_user_df = pd.read_csv(f"user_group_mem/output/group_user {name_suffix}.csv", encoding="utf-8")
    group_user_df["group_users"] = group_user_df["group_users"].apply(eval)

    # Import the main item table
    item_df = pd.read_csv(f"dataset/crossDomainData/user_item_data/{' '.join(domain_list)}/meta_crossdomain.csv", encoding="utf-8")

    # Import interaction table
    inter_df = pd.read_csv(f"dataset/crossDomainData/user_item_data/{' '.join(domain_list)}/timesequence/inter_crossdomain_timesequence_train.csv", encoding="utf-8")
    inter_df = pd.merge(inter_df, item_df[["parent_asin", "title", "main_category", "categories"]], on="parent_asin", how="inner")
    num_rows = len(inter_df)
    inter_df = inter_df.iloc[:int(num_rows * 0.10 * ratio)]

    # Create a txt file for each group and write group memory
    for _, group in group_user_df.iterrows():
        group_name = group["group_name"].replace('"', '')
        group_user_list = group["group_users"]

        # Filter interaction records for this group's users from inter_df
        group_interactions = inter_df[inter_df["user_id"].isin(group_user_list)]

        # Write interactions in group memory
        if not os.path.exists(f"memory/{exp_name}/groupMem"):
            os.makedirs(f"memory/{exp_name}/groupMem")

        with open(f"memory/{exp_name}/groupMem/{group_name}.txt", "w", encoding="utf-8") as file:
            file.write(f"Users who have similar preferences to me in {group_name} have interacted with the following items recently:\n\n")
            domain0_txt = f"{domain_list[0]}:"
            domain1_txt = f"{domain_list[1]}:"
            domain2_txt = f"{domain_list[2]}:"
            if len(domain_list) == 4:
                domain3_txt = f"{domain_list[3]}"

            for _, interaction in group_interactions.iterrows():
                if interaction["main_category"] == get_main_kind(domain_list[0]):
                    domain0_txt += interaction['title'] + ";"
                if interaction["main_category"] == get_main_kind(domain_list[1]):
                    domain1_txt += interaction['title'] + ";"
                if interaction["main_category"] == get_main_kind(domain_list[2]):
                    domain2_txt += interaction['title'] + ";"
                if len(domain_list) == 4 and interaction["main_category"] == get_main_kind(domain_list[3]):
                    domain3_txt += interaction['title'] + ";"

            if len(domain_list) == 4:
                file.write(f"{domain0_txt} \n\n {domain1_txt}\n\n {domain2_txt}\n\n {domain3_txt}")
            else:
                file.write(f"{domain0_txt} \n\n {domain1_txt}\n\n {domain2_txt}")

if __name__ == "__main__":
    exp_name = "AgentCF++ " + ' '.join(domain_list)
    name_suffix = exp_name.replace("AgentCF++ ", "")
    process(exp_name, name_suffix, 10)
