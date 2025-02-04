domain_main_category_dict = {
    "Books": "Books",
    "Movies_and_TV": "Movies & TV",
    "Beauty_and_Personal_Care": "All Beauty",
    "Electronics": "All Electronics",
    "Sports_and_Outdoors": "Sports & Outdoors",
    "CDs_and_Vinyl": "Digital Music",
    "Video_Games": "Video Games"
}

candidate_num = 10
model = "gpt-4o-mini"  # Model type
prompt_strategy = "B"
evaluation_times = 1
# domain_list = ["Books", "CDs_and_Vinyl", "Movies_and_TV"]
# domain_list = ["Video_Games", "CDs_and_Vinyl", "Movies_and_TV"]
domain_list = ["Books", "Video_Games", "Movies_and_TV"]
# domain_list = ["Books", "CDs_and_Vinyl", "Video_Games"]
# domain_list = ["Books", "CDs_and_Vinyl", "Video_Games", "Movies_and_TV"]
n_random_item = 100

def inter_data_source(mode):
    inter_data_source = f"dataset\\crossDomainData\\user_item_data\\{' '.join(domain_list)}\\timesequence\\inter_crossdomain_timesequence_{mode}.csv"
    return inter_data_source

def get_main_kind(domain):
    return domain_main_category_dict[domain]

random_domain0_source = f"dataset\\crossDomainData\\user_item_data\\{' '.join(domain_list)}\\random\\random_{domain_list[0]}.csv"
random_domain1_source = f"dataset\\crossDomainData\\user_item_data\\{' '.join(domain_list)}\\random\\random_{domain_list[1]}.csv"
random_domain2_source = f"dataset\\crossDomainData\\user_item_data\\{' '.join(domain_list)}\\random\\random_{domain_list[2]}.csv"
random_domain3_source = ''
if len(domain_list) == 4:
    random_domain3_source = f"dataset\\crossDomainData\\user_item_data\\{' '.join(domain_list)}\\random\\random_{domain_list[3]}.csv"

item_data_source = f"dataset\\crossDomainData\\user_item_data\\{' '.join(domain_list)}\\meta_crossdomain.csv"

group_Mem_length = 5
group_n_cluster = 384
is_use_intermediate_node = True