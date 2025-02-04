"""
This script is used to group users
"""
import pandas as pd
import json
import ast
from itertools import chain
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt import get_call_llm_for_summary
from request import get_response_from_openai
from config import model, domain_list

"""
Hyperparameters
"""
num_groups = 10

"""
Define a function to get a list of user IDs based on a list of tags
"""
def get_user_ids_by_tags(tags, tag_user_dict):
    user_ids = set()
    for tag in tags:
        if tag in tag_user_dict:
            user_ids.update(tag_user_dict[tag])
    return list(user_ids)

"""
Summarize preferences for each group
"""
def call_llm_for_summary(tag_list):
    prompt = get_call_llm_for_summary(tag_list=tag_list)
    return get_response_from_openai(prompt=prompt, model=model)

def process(exp_name, name_suffix):
    """
    Import user-tag data and tag clustering data
    """
    # Import user-tag data
    with open(f"user_group_mem/llm4embedding/input/user_tag {name_suffix}.json", "r", encoding="utf-8") as file:
        user_tag_dict = json.load(file)
    # Import cluster-tags data
    cluster_tags_df = pd.read_csv(f"user_group_mem/output/cluster_tags {name_suffix}.csv", encoding="utf-8")

    """
    Calculate and select the largest categories containing the most users
    """
    cluster_tags_df['0'] = cluster_tags_df['0'].apply(lambda x: ast.literal_eval(x))
    cluster_tags_df["num_user"] = cluster_tags_df["0"].apply(lambda x: len(x))
    cluster_tags_df = cluster_tags_df.sort_values(by="num_user", ascending=False)
    top_groups = cluster_tags_df.head(num_groups)

    """
    Convert user-tag dictionary to tag-user dictionary
    """
    tag_user_dict = {}
    for user_id, tags in user_tag_dict.items():
        for tag in tags:
            if tag not in tag_user_dict:
                tag_user_dict[tag] = []
            tag_user_dict[tag].append(user_id)

    """
    Assign users to different groups
    """
    top_groups['group_users'] = top_groups['0'].apply(lambda tags: get_user_ids_by_tags(tags, tag_user_dict))
    # Combine all list elements from the DataFrame column into an iterator
    all_elements = chain.from_iterable(top_groups['group_users'])
    # Convert the iterator to a set to remove duplicates
    unique_users = set(all_elements)
    num_unique_users = len(unique_users)

    top_groups["group_name"] = top_groups["0"].apply(call_llm_for_summary)
    top_groups.drop(columns=["0", "cluster", "num_user"], inplace=True)

    top_groups.to_csv(f"user_group_mem/output/group_user {name_suffix}.csv", index=False)

if __name__ == "__main__":
    exp_name = "AgentCF++ " + " ".join(domain_list)
    name_suffix = exp_name.replace("AgentCF++ ", "")
    process(exp_name, name_suffix)