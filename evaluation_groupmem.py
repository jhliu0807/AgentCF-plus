"""
Evaluate experimental effects
"""
import math
from dataPrepare import createInterDF, createItemDF, createRandomDF
from config import candidate_num, model, prompt_strategy, evaluation_times, inter_data_source, item_data_source, random_domain0_source, random_domain1_source, random_domain2_source, get_main_kind, domain_list, group_Mem_length, random_domain3_source
import random
from prompt import system_prompt_template_evaluation_basic_g, system_prompt_template_evaluation_sequential_g, system_prompt_template_evaluation_retrieval_g
import re
import pandas as pd
from fuzzywuzzy import fuzz
from request import get_response_from_openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

exp_name = "AgentCF" + " " + " ".join(domain_list)
name_suffix = exp_name.replace("AgentCF ", "")
group_mem_exp_name = "AgentCF++" + " " + " ".join(domain_list)
mode = "test"
max_retries = 3

def find_most_similar_memory(string_list, target):
    '''
    The current retrieval method uses TF-IDF vector retrieval for the most similar memory; there is room for improvement.
    '''
    # Add the target string to the list for vectorization
    combined_list = string_list + [target]
    # Use TfidfVectorizer to convert text to TF-IDF feature vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_list)
    # Extract the TF-IDF vector of the target string (note the index; the last one is the target string)
    target_vector = tfidf_matrix[-1].reshape(1, -1)
    # Calculate the cosine similarity of all strings with the target string (excluding the target string itself)
    cosine_sims = cosine_similarity(tfidf_matrix[:-1], target_vector)
    # Find the index of the highest similarity
    most_similar_idx = cosine_sims.argmax(axis=0)[0]
    # Return the string with the highest similarity
    return string_list[most_similar_idx]

def calculate_dcg(relevance_scores, k):
    dcg = 0.0
    for i in range(k):  
        if i < len(relevance_scores):  
            dcg += relevance_scores[i] / math.log2(i + 2)  # Note that here i+2 is used because counting starts from 0, but log2(1) is meaningless, so i+2 is commonly used for smoothing
        else:
            break  # Stop if the list length is less than k
    return dcg
  
def calculate_idcg(relevance_scores, k):  
    sorted_scores = sorted(relevance_scores, reverse=True)  # Sort from high to low  
    return calculate_dcg(sorted_scores[:k], k)  # Calculate DCG for the top k as iDCG  
  
def calculate_ndcg(relevance_scores, k):
    dcg_k = calculate_dcg(relevance_scores, k)
    idcg_k = calculate_idcg(relevance_scores, k)
    if idcg_k == 0:  # Avoid division by zero
        return 0.0
    return dcg_k / idcg_k

def get_similarity_score_list(system_evaluation_prompt, model):
    # Get the output from the large model and sort the final results
    responseText = get_response_from_openai(system_evaluation_prompt, model)

    # Final recommendation results and relevance scores
    result_list = []
    similarity_score_list = []
    # Split the text data into a list by line
    lines = responseText.split("Rank:")[-1].splitlines()

    # Iterate through each line and check if it starts with a number
    for line_num, line in enumerate(lines, start=1):  
        if line.strip() and line[0].isdigit():  # Use strip() to remove leading and trailing whitespace, check if the first character is a number
            temp_title = re.split('\d\.', line)[-1].strip()
            result_list.append(temp_title)
            similarity_score_list.append(fuzz.ratio(temp_title.lower(), target_item_title.lower()))
    return similarity_score_list

if __name__ == "__main__":
    # Construct three large tables
    interDF = createInterDF(inter_data_source(mode))
    itemDF = createItemDF(item_data_source)

    # Build random selection datasets
    random_domain0_DF = createRandomDF(random_domain0_source)
    random_domain1_DF = createRandomDF(random_domain1_source)
    random_domain2_DF = createRandomDF(random_domain2_source)
    if len(domain_list) == 4:
        random_domain3_DF = createRandomDF(random_domain3_source)

    ndcg_10_list = []
    ndcg_5_list = []
    ndcg_1_list = []
    mrr_list = []
    for index, record in interDF.iterrows():
        target_itemId = record["parent_asin"]
        userId = record["user_id"]
        target_item_title = str(itemDF[itemDF["parent_asin"] == target_itemId]["title"].values[0])

        # Read user memory
        with open(f".\\memory\\{exp_name}\\user\\user.{userId}", "r", encoding="utf-8") as file:
            user_memory = file.read()

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
        
        # Randomly shuffle data
        random.shuffle(random_itemId_list)

        # Read candidate-related information
        cdt_item_memory_list = []
        cdt_item_title_list = []
        for cdt_itemId in random_itemId_list:
            try:
                with open(f".\\memory\\{exp_name}\\item\\item.{cdt_itemId}", "r", encoding="utf-8") as file:
                    cdt_item_memory_list.append(file.read())
                cdt_item_title_list.append(str(itemDF[itemDF["parent_asin"] == cdt_itemId]["title"].values[0]))
            except Exception as e:
                # Print error message and continue the loop if any exception occurs
                print(f"Error processing item {cdt_itemId}: {e}")
                cdt_item_memory_list.append('nan')
                cdt_item_title_list.append('nan')
                continue

        # Construct user description and item description
        user_description = user_memory
        example_list_of_item_description = ''
        for cdt_item_memory, cdt_item_title in zip(cdt_item_memory_list, cdt_item_title_list):
            example_list_of_item_description += f"title:{cdt_item_title.strip()}. description:{cdt_item_memory.strip()}\n"

        # Choose the appropriate prompt strategy
        if prompt_strategy == "B":
            system_evaluation_prompt = system_prompt_template_evaluation_basic_g(user_description, candidate_num, example_list_of_item_description)
        elif prompt_strategy == "B+H":
            historical_inter_itemId_list = list(interDF[interDF['user_id'] == userId]["parent_asin"])  # Select historical interacted items
            historical_inter_itemId_list = [x for x in historical_inter_itemId_list if x != target_itemId]  # Filter out the current target
            historical_inter_item_memory_list = []
            historical_inter_item_title_list = []
            historical_interactions = ""
            for historical_inter_itemId in historical_inter_itemId_list:
                with open(f".\\memory\\{exp_name}\\item\\item.{historical_inter_itemId}", "r", encoding="utf-8") as file:
                    historical_inter_item_memory_list.append(file.read())
                historical_inter_item_title_list.append(str(itemDF[itemDF["parent_asin"] == historical_inter_itemId]["title"].values[0]))

            for historical_inter_item_memory, historical_inter_item_title in zip(historical_inter_item_memory_list, historical_inter_item_title_list):
                historical_interactions += f"title:{historical_inter_item_title.strip()}. description:{historical_inter_item_memory.strip()}\n"
            system_evaluation_prompt = system_prompt_template_evaluation_sequential_g(user_description, historical_interactions, candidate_num, example_list_of_item_description)

        elif prompt_strategy == "B+R":  # Use retrieval from long-term memory as the prompt strategy
            # Read user_long_memory
            with open(f".\\memory\\{exp_name}\\user-long\\user.{userId}", "r", encoding="utf-8") as file:
                user_long_memory = file.read()
            user_long_memory_list = user_long_memory.split("\n=====\n")[:-1]  # Exclude current short-term memory
            if len(user_long_memory_list) == 0:
                user_long_memory_list.append(user_long_memory.split("\n=====\n")[-1])
            cdt_retrieval_prompt = " ".join(cdt_item_memory_list)
            most_similar_user_memory = find_most_similar_memory(user_long_memory_list, cdt_retrieval_prompt)
            system_evaluation_prompt = system_prompt_template_evaluation_retrieval_g(most_similar_user_memory, user_description, candidate_num, example_list_of_item_description)        

        # Sort multiple times to reduce randomness
        for i in range(evaluation_times):
            # Get similarity score list
            similarity_score_list = get_similarity_score_list(system_evaluation_prompt, model)
            retries = 0
            # If the similarity score list has issues, regenerate
            while len(similarity_score_list) != candidate_num and retries < max_retries:
                retries += 1
                print(f"retry {retries} ...")
                similarity_score_list = get_similarity_score_list(system_evaluation_prompt, model)

            relevance_score_list = [1 if x == max(similarity_score_list) else 0 for x in similarity_score_list]
            target_rank = relevance_score_list.index(1) + 1  # Find the ranking of the target
            ndcg_at_10 = calculate_ndcg(relevance_score_list, 10)
            ndcg_10_list.append(ndcg_at_10)
            ndcg_at_5 = calculate_ndcg(relevance_score_list, 5)
            ndcg_5_list.append(ndcg_at_5)
            ndcg_at_1 = calculate_ndcg(relevance_score_list, 1)
            ndcg_1_list.append(ndcg_at_1)

            mrr_list.append(1.0 / target_rank)
            print(f"ndcg@10: {ndcg_at_10} mean: {sum(ndcg_10_list) / len(ndcg_10_list)}")
            print(f"ndcg@5: {ndcg_at_5} mean: {sum(ndcg_5_list) / len(ndcg_5_list)}")
            print(f"ndcg@1: {ndcg_at_1} mean: {sum(ndcg_1_list) / len(ndcg_1_list)}")
            print(f"1/rank: {1.0 / target_rank} mrr: {sum(mrr_list) / len(mrr_list)}")

    with open(".\\log\\result.txt", mode="a", encoding="utf-8") as file:
        file.write(f"{exp_name}:\nPrompt strategy: {prompt_strategy}\nNDCG@10:  {sum(ndcg_10_list) / len(ndcg_10_list)}\nNDCG@5:  {sum(ndcg_5_list) / len(ndcg_5_list)}\nNDCG@1:  {sum(ndcg_1_list) / len(ndcg_1_list)}\nMRR:  {sum(mrr_list) / len(mrr_list)}\n\n")