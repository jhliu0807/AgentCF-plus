import pandas as pd
import os
import shutil
from config import inter_data_source, item_data_source, random_domain0_source, random_domain1_source, random_domain2_source, random_domain3_source

def createRandomDF(file_path):
    return pd.read_csv(file_path, dtype=str)

def createItemDF(file_path):
    '''
    Build a complete table of item information
    '''
    return pd.read_csv(file_path)

def createInterDF(file_path):
    '''
    Build interaction dataset
    '''
    return pd.read_csv(file_path)

def prepare_data_from_interDF(mode, domain_list, crossDomain):
    '''
    Collect users and items from interDF
    Build the dataset required for the experiment through interaction information
    '''
    interDF = createInterDF(inter_data_source(mode))
    itemDF = createItemDF(item_data_source)

    if not crossDomain:
        # Rename memory files to user and item IDs for easier maintenance
        _ = []
        for i in range(interDF.shape[0]):
            for j in range(1, interDF.shape[1]):  
                # Access the value of the current element (note: using .iloc, which is index-based)
                element_value = interDF.iloc[i, j]
                _.append(element_value)
        itemList = list(dict.fromkeys(_))
    else:
        itemList = set(interDF["parent_asin"])

    # Initialize item memory
    for itemId in itemList:
        item_main_category = itemDF[itemDF['parent_asin'] == itemId]['main_category'].values[0]
        item_title = itemDF[itemDF['parent_asin'] == itemId]['title'].values[0]
        item_subtitle = itemDF[itemDF['parent_asin'] == itemId]['subtitle'].values[0]
        item_class = itemDF[itemDF['parent_asin'] == itemId]['categories'].values[0]
        item_price = itemDF[itemDF['parent_asin'] == itemId]['price'].values[0]

        init_item_memory = f"'main_category':{item_main_category}, 'item_title': '{item_title}', 'item_subtitle': '{item_subtitle}', 'item_class': '{item_class}', 'item_price': '{item_price}'"
        
        output_dir = f".\\dataset\\crossDomainData\\initial\\{' '.join(domain_list)}\\item"
        os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist
        with open(f".\\dataset\\crossDomainData\\initial\\{' '.join(domain_list)}\\item\\item.{itemId}", "w", encoding="utf-8") as file:
            file.write(init_item_memory)

    # Initialize user memory
    for userId in interDF['user_id'].values:
        if len(domain_list) == 4:
            init_user_memory = f"I enjoy {domain_list[0]}, {domain_list[1]}, {domain_list[2]} and {domain_list[3]} very much."
        else:
            init_user_memory = f"I enjoy {domain_list[0]}, {domain_list[1]} and {domain_list[2]} very much."

        output_dir = f".\\dataset\\crossDomainData\\initial\\{' '.join(domain_list)}\\user"
        os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist
        with open(f".\\dataset\\crossDomainData\\initial\\{' '.join(domain_list)}\\user\\user.{userId}", "w", encoding="utf-8") as file:
            file.write(init_user_memory)
    
    shutil.copytree(f"dataset\\crossDomainData\\initial\\{' '.join(domain_list)}\\user", f"dataset\\crossDomainData\\initial\\{' '.join(domain_list)}\\user-long")

def prepare_initial_mem_from_interDF(mode, domain_list):
    '''
    Collect users and items from interDF
    Build the dataset required for the experiment through interaction information
    '''
    interDF = createInterDF(inter_data_source(mode))
    itemDF = createItemDF(item_data_source)

    random_domain0_df = createRandomDF(random_domain0_source)
    random_domain1_df = createRandomDF(random_domain1_source)
    random_domain2_df = createRandomDF(random_domain2_source)
    if len(domain_list) == 4:
        random_domain3_df = createRandomDF(random_domain3_source)

    itemList = set(interDF["parent_asin"])

    # Initialize item memory
    for itemId in itemList:
        item_main_category = itemDF[itemDF['parent_asin'] == itemId]['main_category'].values[0]
        item_title = itemDF[itemDF['parent_asin'] == itemId]['title'].values[0]
        item_subtitle = itemDF[itemDF['parent_asin'] == itemId]['subtitle'].values[0]
        item_class = itemDF[itemDF['parent_asin'] == itemId]['categories'].values[0]
        item_price = itemDF[itemDF['parent_asin'] == itemId]['price'].values[0]

        init_item_memory = f"'main_category':{item_main_category}, 'item_title': '{item_title}', 'item_subtitle': '{item_subtitle}', 'item_class': '{item_class}', 'item_price': '{item_price}'"
        
        output_dir = f".\\dataset\\crossDomainData\\initial\\{' '.join(domain_list)}\\AgentCF++\\item"
        os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist
        with open(f".\\dataset\\crossDomainData\\initial\\{' '.join(domain_list)}\\AgentCF++\\item\\item.{itemId}", "w", encoding="utf-8") as file:
            file.write(init_item_memory)

    # Determine which domains each user interacted with
    user_domains = {}
    # Iterate through each user
    for index, row in random_domain0_df.iterrows():
        user_id = row['Unnamed: 0']  # Get user ID
        item_types = []
        # Check df1
        if not row.iloc[1:].eq('0').all():  # Check from the second column
            item_types.append(domain_list[0])
        # Check df2
        if not random_domain1_df.loc[random_domain1_df['Unnamed: 0'] == user_id].iloc[0, 1:].eq('0').all():
            item_types.append(domain_list[1])
        # Check df3
        if not random_domain2_df.loc[random_domain2_df['Unnamed: 0'] == user_id].iloc[0, 1:].eq('0').all():
            item_types.append(domain_list[2])
        if len(domain_list) == 4 and not random_domain3_df.loc[random_domain3_df['Unnamed: 0'] == user_id].iloc[0, 1:].eq('0').all():
            item_types.append(domain_list[3])
        # Store the item types that the user interacted with
        user_domains[user_id] = item_types

    # Initialize user memory
    for userId in interDF['user_id'].values:
        if not os.path.exists(f".\\dataset\\crossDomainData\\initial\\{' '.join(domain_list)}\\AgentCF++\\user\\user.{userId}"):
            os.makedirs(f".\\dataset\\crossDomainData\\initial\\{' '.join(domain_list)}\\AgentCF++\\user\\user.{userId}")

        if domain_list[0] in user_domains[userId]:
            init_user_memory_domain0 = f"I am an Amazon buyer, and I enjoy {domain_list[0]} very much."
            with open(f".\\dataset\\crossDomainData\\initial\\{' '.join(domain_list)}\\AgentCF++\\user\\user.{userId}\\private-{domain_list[0]}.txt", "w", encoding="utf-8") as file:
                file.write(init_user_memory_domain0)
            with open(f".\\dataset\\crossDomainData\\initial\\{' '.join(domain_list)}\\AgentCF++\\user\\user.{userId}\\crossDomain-{domain_list[0]}.txt", "w", encoding="utf-8") as file:
                file.write(init_user_memory_domain0)

        if domain_list[1] in user_domains[userId]:
            init_user_memory_domain1 = f"I am an Amazon buyer, and I enjoy {domain_list[1]} very much."
            with open(f".\\dataset\\crossDomainData\\initial\\{' '.join(domain_list)}\\AgentCF++\\user\\user.{userId}\\private-{domain_list[1]}.txt", "w", encoding="utf-8") as file:
                file.write(init_user_memory_domain1)
            with open(f".\\dataset\\crossDomainData\\initial\\{' '.join(domain_list)}\\AgentCF++\\user\\user.{userId}\\crossDomain-{domain_list[1]}.txt", "w", encoding="utf-8") as file:
                file.write(init_user_memory_domain1)

        if domain_list[2] in user_domains[userId]:
            init_user_memory_domain2 = f"I am an Amazon buyer, and I enjoy {domain_list[2]} very much."
            with open(f".\\dataset\\crossDomainData\\initial\\{' '.join(domain_list)}\\AgentCF++\\user\\user.{userId}\\private-{domain_list[2]}.txt", "w", encoding="utf-8") as file:
                file.write(init_user_memory_domain2)
            with open(f".\\dataset\\crossDomainData\\initial\\{' '.join(domain_list)}\\AgentCF++\\user\\user.{userId}\\crossDomain-{domain_list[2]}.txt", "w", encoding="utf-8") as file:
                file.write(init_user_memory_domain2)
            
        if len(domain_list) == 4 and domain_list[3] in user_domains[userId]:
            init_user_memory_domain3 = f"I am an Amazon buyer, and I enjoy {domain_list[3]} very much."
            with open(f".\\dataset\\crossDomainData\\initial\\{' '.join(domain_list)}\\AgentCF++\\user\\user.{userId}\\private-{domain_list[3]}.txt", "w", encoding="utf-8") as file:
                file.write(init_user_memory_domain3)
            with open(f".\\dataset\\crossDomainData\\initial\\{' '.join(domain_list)}\\AgentCF++\\user\\user.{userId}\\crossDomain-{domain_list[3]}.txt", "w", encoding="utf-8") as file:
                file.write(init_user_memory_domain3)