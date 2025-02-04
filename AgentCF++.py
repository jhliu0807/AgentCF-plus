from prompt import *
import random
import re
from fuzzywuzzy import fuzz
import shutil
import os
from dataPrepare import createInterDF, createItemDF, createRandomDF
from config import model, cross_domain, inter_data_source, random_domain0_source, random_domain1_source, random_domain2_source, item_data_source, domain_list, get_main_kind, random_domain3_source
from request import get_response_from_openai
from functions import concatenate_crossdomain_preference
from tqdm import tqdm
import pandas as pd

exp_name = "AgentCF++" + " " + " ".join(domain_list)
mode = "train"

def initialize_memory(exp_name, domain_list):
    '''
    Initialize user and item memory (directly copy from the saved initial memory)
    '''
    memory_dir = f".\\memory\\{exp_name}"  # Memory directory
    if os.path.exists(os.path.join(memory_dir, "item")) or os.path.exists(os.path.join(memory_dir, "user")):
        exit()
        shutil.rmtree(os.path.join(memory_dir, "item"))
        shutil.rmtree(os.path.join(memory_dir, "user"))
    shutil.copytree(f"dataset\\crossDomainData\\initial\\{' '.join(domain_list)}\\AgentCF++\\item", os.path.join(memory_dir, "item"))
    shutil.copytree(f"dataset\\crossDomainData\\initial\\{' '.join(domain_list)}\\AgentCF++\\user", os.path.join(memory_dir, "user"))

def save_memory(ratio):
    src_folder = f"memory\\{exp_name}"
    dst_folder = f"memory\\{exp_name + '_' + ratio}"
    try:
        shutil.copytree(src_folder, dst_folder)
        print(f"Folder '{src_folder}' successfully copied to '{dst_folder}'")
    except Exception as e:
        print(f"Error copying folder: {e}")

def save_old_memory(userId, pos_itemId, main_kind, single_domain_memory, cross_domain_preference, pos_item_memory):
    additional_text = '===before update===\n'
    # Additional text to append
    additional_text += "userid:" + userId + "\n"
    additional_text += "itemid:" + pos_itemId + "\n"
    additional_text += "main_kind:" + main_kind + "\n"
    additional_text += "single_domain_memory:\n" + single_domain_memory + "\n"
    additional_text += "cross_domain_preference:\n" + cross_domain_preference + "\n"
    additional_text += "item_memory:\n" + pos_item_memory + "\n\n"
    with open('log\\case_study.txt', 'a', encoding='utf-8') as file:
        file.write(additional_text)

def save_new_memory(userId, pos_itemId, main_kind, single_domain_memory, private_domain_description, cross_domain_preference, pos_item_memory):
    additional_text = '===after update===\n'
    # Additional text to append
    additional_text += "userid:" + userId + "\n"
    additional_text += "itemid:" + pos_itemId + "\n"
    additional_text += "main_kind:" + main_kind + "\n"
    additional_text += "single_domain_memory:\n" + single_domain_memory + "\n"
    additional_text += "private_domain_description:\n" + private_domain_description + "\n"
    additional_text += "cross_domain_preference:\n" + cross_domain_preference + "\n"
    additional_text += "item_memory:\n" + pos_item_memory + "\n\n"
    with open('log\\case_study.txt', 'a', encoding='utf-8') as file:
        file.write(additional_text)

    return

def process_interaction(interDF, itemDF, random_domain0_DF, random_domain1_DF, random_domain2_DF, random_domain3_DF, exp_name, model, domain_list):
    """
    Start interaction
    """
    all_inter_num = interDF.shape[0]
    save_interval = int(all_inter_num * 0.1)
    for index, record in tqdm(interDF.iterrows()):
        try:
            # Save intermediate results every 10%
            if index % save_interval == 0 and index != 0:
                save_memory(str(int(index / save_interval)))  # Call save function
            pos_itemId = record["parent_asin"]
            userId = record["user_id"]

            with open(f".\\memory\\{exp_name}\\item\\item.{pos_itemId}", "r", encoding="utf-8") as file:
                pos_item_memory = file.read()

            # Extract the name of the positive item
            pos_item_title = itemDF[itemDF["parent_asin"] == pos_itemId]["title"].values[0]

            # Extract the main category of the currently interacted item
            main_kind = itemDF[itemDF["parent_asin"] == pos_itemId]["main_category"].values[0]
            if len(domain_list) == 4:
                neg_itemId = get_neg_item_id_4domains(main_kind, userId, random_domain0_DF, random_domain1_DF, random_domain2_DF, random_domain3_DF, domain_list)
            else:
                neg_itemId = get_neg_item_id(main_kind, userId, random_domain0_DF, random_domain1_DF, random_domain2_DF, domain_list)

            with open(f".\\memory\\{exp_name}\\item\\item.{neg_itemId}", "r", encoding="utf-8") as file:
                neg_item_memory = file.read()
            neg_item_title = itemDF[itemDF["parent_asin"] == neg_itemId]["title"].values[0]

            if main_kind == get_main_kind(domain_list[0]):
                main_kind = domain_list[0]
                with open(f".\\memory\\{exp_name}\\user\\user.{userId}\\private-{domain_list[0]}.txt", "r", encoding="utf-8") as file:
                    single_domain_memory = file.read()
                with open(f".\\memory\\{exp_name}\\user\\user.{userId}\\crossDomain-{domain_list[0]}.txt", "r", encoding="utf-8") as file:
                    cross_domain_preference = file.read()
            elif main_kind == get_main_kind(domain_list[1]):
                main_kind = domain_list[1]
                with open(f".\\memory\\{exp_name}\\user\\user.{userId}\\private-{domain_list[1]}.txt", "r", encoding="utf-8") as file:
                    single_domain_memory = file.read()
                with open(f".\\memory\\{exp_name}\\user\\user.{userId}\\crossDomain-{domain_list[1]}.txt", "r", encoding="utf-8") as file:
                    cross_domain_preference = file.read()
            elif main_kind == get_main_kind(domain_list[2]):
                main_kind = domain_list[2]
                with open(f".\\memory\\{exp_name}\\user\\user.{userId}\\private-{domain_list[2]}.txt", "r", encoding="utf-8") as file:
                    single_domain_memory = file.read()
                with open(f".\\memory\\{exp_name}\\user\\user.{userId}\\crossDomain-{domain_list[2]}.txt", "r", encoding="utf-8") as file:
                    cross_domain_preference = file.read()
            elif len(domain_list) == 4 and main_kind == get_main_kind(domain_list[3]):
                main_kind = domain_list[3]
                with open(f".\\memory\\{exp_name}\\user\\user.{userId}\\private-{domain_list[3]}.txt", "r", encoding="utf-8") as file:
                    single_domain_memory = file.read()
                with open(f".\\memory\\{exp_name}\\user\\user.{userId}\\crossDomain-{domain_list[3]}.txt", "r", encoding="utf-8") as file:
                    cross_domain_preference = file.read()

            save_old_memory(userId, pos_itemId, main_kind, single_domain_memory, cross_domain_preference, pos_item_memory)

            # Construct user description and item description
            user_description = single_domain_memory
            user_description_all = f"My preferences in the type of goods in {main_kind}: " + single_domain_memory + "\n" + "Moreover, " + cross_domain_preference
            list_of_item_description = f"title:{neg_item_title.strip()}. description:{neg_item_memory.strip()}\ntitle:{pos_item_title}. description:{pos_item_memory.strip()}"
            system_prompt = system_prompt_template(cross_domain_preference, list_of_item_description)
            
            # Get output from the large model: which item to choose + explanation
            responseText = get_response_from_openai(system_prompt, model)
            selected_item_title, system_reason = parse_response(responseText)

            # Determine which item the model chose and whether the choice was correct
            pos_similarity = fuzz.ratio(selected_item_title.lower(), pos_item_title.lower())
            neg_similarity = fuzz.ratio(selected_item_title.lower(), neg_item_title.lower())
            is_choice_right = pos_similarity > neg_similarity

            # Create backward prompt for user
            user_prompt = create_user_prompt(user_description, list_of_item_description, pos_item_title, neg_item_title, system_reason, is_choice_right)

            # Get output from the large model: updated self-introduction for the user
            responseText = get_response_from_openai(user_prompt, model)
            new_single_memory = update_user_memory(userId, exp_name, responseText, main_kind)

            private_domain_description = concatenate_crossdomain_preference(f"memory\\{exp_name}\\user\\user.{userId}")
            cross_domain_prompt = system_prompt_crossdomain(cross_domain_preference, private_domain_description, main_kind)
            responseText = get_response_from_openai(cross_domain_prompt, model)
            new_cross_domain_preference = update_user_crossdomain_memory(userId, exp_name, responseText, main_kind)

            if main_kind == domain_list[0]:
                with open(f".\\memory\\{exp_name}\\user\\user.{userId}\\crossDomain-{domain_list[0]}.txt", "r", encoding="utf-8") as file:
                    cross_domain_preference = file.read()
            elif main_kind == domain_list[1]:
                with open(f".\\memory\\{exp_name}\\user\\user.{userId}\\crossDomain-{domain_list[1]}.txt", "r", encoding="utf-8") as file:
                    cross_domain_preference = file.read()
            elif main_kind == domain_list[2]:
                with open(f".\\memory\\{exp_name}\\user\\user.{userId}\\crossDomain-{domain_list[2]}.txt", "r", encoding="utf-8") as file:
                    cross_domain_preference = file.read()
            elif len(domain_list) == 4 and main_kind == domain_list[3]:
                with open(f".\\memory\\{exp_name}\\user\\user.{userId}\\crossDomain-{domain_list[3]}.txt", "r", encoding="utf-8") as file:
                    cross_domain_preference = file.read()

            item_prompt = create_item_prompt(cross_domain_preference, list_of_item_description, pos_item_title, neg_item_title, system_reason, is_choice_right)
            # Get output from the large model: updated information for the item
            responseText = get_response_from_openai(item_prompt, model)
            new_positive_item_memory = update_item_memory(pos_itemId, neg_itemId, exp_name, responseText)

            save_new_memory(userId, pos_itemId, main_kind, new_single_memory, private_domain_description, new_cross_domain_preference, new_positive_item_memory)
            print("\n" + userId + " " + pos_itemId + " already done.")

        except Exception as e:
            print(f"Error processing interaction for user {userId} and item {pos_itemId}: {e}")
            continue

# TODO: Parse different domains
def get_neg_item_id(main_kind, userId, random_domain0_DF, random_domain1_DF, random_domain2_DF, domain_list):
    main_kind = str(main_kind).strip()
    if main_kind == get_main_kind(domain_list[0]):
        df = random_domain0_DF
    elif main_kind == get_main_kind(domain_list[1]):
        df = random_domain1_DF
    elif main_kind == get_main_kind(domain_list[2]):
        df = random_domain2_DF
    else:
        raise ValueError(f"Unknown main kind: {main_kind}")
    return df.loc[df["Unnamed: 0"] == userId, f"item_{random.randint(0, 99)}"].values[0]

def get_neg_item_id_4domains(main_kind, userId, random_domain0_DF, random_domain1_DF, random_domain2_DF, random_domain3_DF, domain_list):
    main_kind = str(main_kind).strip()
    if main_kind == get_main_kind(domain_list[0]):
        df = random_domain0_DF
    elif main_kind == get_main_kind(domain_list[1]):
        df = random_domain1_DF
    elif main_kind == get_main_kind(domain_list[2]):
        df = random_domain2_DF
    elif main_kind == get_main_kind(domain_list[3]):
        df = random_domain3_DF
    else:
        raise ValueError(f"Unknown main kind: {main_kind}")
    return df.loc[df["Unnamed: 0"] == userId, f"item_{random.randint(0, 99)}"].values[0]

def parse_response(responseText):
    selected_item_title = re.split(r"Choice:|\n", responseText)[1]
    system_reason = re.split(r"Explanation:", responseText)[-1].strip()
    return selected_item_title, system_reason

def create_user_prompt(user_description, list_of_item_description, pos_item_title, neg_item_title, system_reason, is_choice_right):
    if not is_choice_right:
        user_prompt = user_prompt_system_role(user_description) + '\n' + user_prompt_template(list_of_item_description, pos_item_title, neg_item_title, system_reason)
    else:
        user_prompt = user_prompt_system_role(user_description) + '\n' + user_prompt_template_true(list_of_item_description, pos_item_title, neg_item_title, system_reason)
    return user_prompt

def create_item_prompt(cross_domain_preference, list_of_item_description, pos_item_title, neg_item_title, system_reason, is_choice_right):
    if not is_choice_right:
        item_prompt = item_prompt_template(cross_domain_preference, list_of_item_description, pos_item_title, neg_item_title, system_reason)
    else:
        item_prompt = item_prompt_template_true(cross_domain_preference, list_of_item_description, pos_item_title, neg_item_title)
    return item_prompt

def update_user_memory(userId, exp_name, responseText, main_kind):
    responseText = responseText.split("My updated self-introduction:")[-1].strip()
    user_memory_path = os.path.join(f".\\memory\\{exp_name}\\user", f"user.{userId}", f"private-{main_kind}.txt")
    with open(user_memory_path, "w", encoding="utf-8") as file:
        file.write(responseText)
    return responseText

def update_user_crossdomain_memory(userId, exp_name, responseText, main_kind):
    responseText = responseText.split("My deduced preference:")[-1].strip()
    cross_domain_memory_path = os.path.join(f".\\memory\\{exp_name}\\user", f"user.{userId}", f"crossDomain-{main_kind}.txt")
    with open(cross_domain_memory_path, "w", encoding="utf-8") as file:
        file.write(responseText)
    return responseText
 
def update_item_memory(pos_itemId, neg_itemId, exp_name, responseText):
    updated_pos_item_intro = responseText.split("The updated description of the second item is: ")[-1]
    updated_neg_item_intro = re.split(r"The updated description of the first item is: |The updated description of the second item is: ", responseText)[1]
    # Update the item's self-description
    with open(f".\\memory\\{exp_name}\\item\\item.{pos_itemId}", "w", encoding="utf-8") as file:
        file.write(updated_pos_item_intro)
    with open(f".\\memory\\{exp_name}\\item\\item.{neg_itemId}", "w", encoding="utf-8") as file:
        file.write(updated_neg_item_intro)
    return updated_pos_item_intro


if __name__ == "__main__":
    # Build interaction dataset
    interDF = createInterDF(inter_data_source(mode), crossDomain=cross_domain)
    # Build random selection dataset
    random_domain0_DF = createRandomDF(random_domain0_source, crossDomain=cross_domain)
    random_domain1_DF = createRandomDF(random_domain1_source, crossDomain=cross_domain)
    random_domain2_DF = createRandomDF(random_domain2_source, crossDomain=cross_domain)
    random_domain3_DF = pd.DataFrame()
    if len(domain_list) == 4:
        random_domain3_DF = createRandomDF(random_domain3_source, crossDomain=cross_domain)
    # Build the complete item information table
    itemDF = createItemDF(item_data_source, crossDomain=cross_domain)

    initialize_memory(exp_name, domain_list)
    process_interaction(interDF, itemDF, random_domain0_DF, random_domain1_DF, random_domain2_DF, random_domain3_DF, exp_name, model, domain_list)