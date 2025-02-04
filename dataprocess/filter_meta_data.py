import json
import pandas as pd

def filter_books_meta_data(wanted_keys):
    '''
    Filter book data through crossDomainData
    '''
    # Collect all item IDs
    inter_Books_df = pd.read_csv("dataset/crossDomainData/org_data/inter/Books.csv", encoding="utf-8")
    Books_parent_asin_set = set(inter_Books_df['parent_asin'].unique())

    # Filter item IDs and export to CSV
    meta_books_data = []
    with open('dataset/crossDomainData/org_data/meta/meta_Books.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            book_data = json.loads(line.strip())
            if book_data.get('parent_asin') in Books_parent_asin_set and str(book_data.get("title")) != "nan" and book_data.get("title") and str(book_data.get("main_category")) != "nan" and book_data.get("main_category"):
                if "author" in book_data.keys() and book_data["author"] and "name" in book_data["author"].keys():
                    book_data["author"] = book_data["author"]["name"]
                else:
                    book_data["author"] = None
                if book_data["categories"]:
                    book_data["categories"] = ";".join(book_data["categories"])
                book_data = {key: value for key, value in book_data.items() if key in wanted_keys}
                meta_books_data.append(book_data)
    
    # Convert filtered data to DataFrame
    Books_filtered_df = pd.DataFrame(meta_books_data)
    Books_filtered_df["main_category"] = "Books"
    Books_filtered_df.to_csv('dataset/crossDomainData/filtered_data/meta/meta_Books.csv', index=False)

def filter_movies_meta_data(wanted_keys):
    '''
    Filter movie data through crossDomainData
    '''
    # Collect all item IDs
    inter_Movies_df = pd.read_csv("dataset/crossDomainData/org_data/inter/Movies_and_TV.csv", encoding="utf-8")
    Movies_parent_asin_set = set(inter_Movies_df['parent_asin'].unique())

    # Filter item IDs and export to CSV
    meta_movies_data = []
    with open('dataset/crossDomainData/org_data/meta/meta_Movies_and_TV.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            movie_data = json.loads(line.strip())
            if movie_data.get('parent_asin') in Movies_parent_asin_set and str(movie_data.get("title")) != "nan" and movie_data.get("title") and str(movie_data.get("main_category")) != "nan" and movie_data.get("main_category"):
                if movie_data["categories"]:
                    movie_data["categories"] = ";".join(movie_data["categories"])
                movie_data = {key: value for key, value in movie_data.items() if key in wanted_keys}
                meta_movies_data.append(movie_data)
    
    # Convert filtered data to DataFrame
    Movies_filtered_df = pd.DataFrame(meta_movies_data)
    Movies_filtered_df["main_category"] = "Movies & TV"
    Movies_filtered_df.to_csv('dataset/crossDomainData/filtered_data/meta/meta_Movies_and_TV.csv', index=False)

def filter_beauty_meta_data(wanted_keys):
    '''
    Filter beauty data through crossDomainData
    '''
    inter_beauty_df = pd.read_csv("dataset/crossDomainData/org_data/inter/Beauty_and_Personal_Care.csv", encoding="utf-8")
    beauty_parent_asin_set = set(inter_beauty_df["parent_asin"].unique())

    # Filter item IDs and export to CSV
    meta_beauty_data = []
    with open("dataset/crossDomainData/org_data/meta/meta_Beauty_and_Personal_Care.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            beauty_data = json.loads(line.strip())
            if beauty_data.get("parent_asin") in beauty_parent_asin_set and str(beauty_data.get("title")) != "nan" and beauty_data.get("title") and str(beauty_data.get("main_category")) != "nan" and beauty_data.get("main_category"):
                if beauty_data["categories"]:
                    beauty_data["categories"] = ";".join(beauty_data["categories"])
                beauty_data = {key: value for key, value in beauty_data.items() if key in wanted_keys}
                meta_beauty_data.append(beauty_data)

    Beauty_filtered_df = pd.DataFrame(meta_beauty_data)
    Beauty_filtered_df["main_category"] = "All Beauty"
    Beauty_filtered_df.to_csv("dataset/crossDomainData/filtered_data/meta/meta_Beauty_and_Personal_Care.csv", index=False)

def filter_electronics_meta_data(wanted_keys):
    '''
    Filter electronics data through crossDomainData
    '''
    inter_electronics_df = pd.read_csv("dataset/crossDomainData/org_data/inter/Electronics.csv", encoding="utf-8")
    electronics_parent_asin_set = set(inter_electronics_df["parent_asin"].unique())

    # Filter item IDs and export to CSV
    meta_electronics_data = []
    with open("dataset/crossDomainData/org_data/meta/meta_Electronics.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            electronics_data = json.loads(line.strip())
            if electronics_data.get("parent_asin") in electronics_parent_asin_set and str(electronics_data.get("title")) != "nan" and electronics_data.get("title") and str(electronics_data.get("main_category")) != "nan" and electronics_data.get("main_category"):
                if electronics_data["categories"]:
                    electronics_data["categories"] = ";".join(electronics_data["categories"])
                electronics_data = {key: value for key, value in electronics_data.items() if key in wanted_keys}
                meta_electronics_data.append(electronics_data)

    Electronics_filtered_df = pd.DataFrame(meta_electronics_data)
    Electronics_filtered_df["main_category"] = "All Electronics"
    Electronics_filtered_df.to_csv("dataset/crossDomainData/filtered_data/meta/meta_Electronics.csv", index=False)

def filter_sports_meta_data(wanted_keys):
    '''
    Filter sports and outdoors data through crossDomainData
    '''
    inter_sports_df = pd.read_csv("dataset/crossDomainData/org_data/inter/Sports_and_Outdoors.csv", encoding="utf-8")
    sports_parent_asin_set = set(inter_sports_df["parent_asin"].unique())

    # Filter item IDs and export to CSV
    meta_sports_data = []
    with open("dataset/crossDomainData/org_data/meta/meta_Sports_and_Outdoors.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            sports_data = json.loads(line.strip())
            if sports_data.get("parent_asin") in sports_parent_asin_set and str(sports_data.get("title")) != "nan" and sports_data.get("title") and str(sports_data.get("main_category")) != "nan" and sports_data.get("main_category"):
                if sports_data["categories"]:
                    sports_data["categories"] = ";".join(sports_data["categories"])
                sports_data = {key: value for key, value in sports_data.items() if key in wanted_keys}
                meta_sports_data.append(sports_data)

    Sports_filtered_df = pd.DataFrame(meta_sports_data)
    Sports_filtered_df["main_category"] = "Sports & Outdoors"
    Sports_filtered_df.to_csv("dataset/crossDomainData/filtered_data/meta/meta_Sports_and_Outdoors.csv", index=False)

def filter_cds_meta_data(wanted_keys):
    '''
    Filter CDs and Vinyl (CDs) data through crossDomainData
    '''
    inter_cds_df = pd.read_csv("dataset/crossDomainData/org_data/inter/CDs_and_Vinyl.csv", encoding="utf-8")
    cds_parent_asin_set = set(inter_cds_df["parent_asin"].unique())

    # Filter item IDs and export to CSV
    meta_cds_data = []
    with open("dataset/crossDomainData/org_data/meta/meta_CDs_and_Vinyl.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            cds_data = json.loads(line.strip())
            if cds_data.get("parent_asin") in cds_parent_asin_set and str(cds_data.get("title")) != "nan" and cds_data.get("title") and str(cds_data.get("main_category")) != "nan" and cds_data.get("main_category"):
                if cds_data["categories"]:
                    cds_data["categories"] = ";".join(cds_data["categories"])
                cds_data = {key: value for key, value in cds_data.items() if key in wanted_keys}
                meta_cds_data.append(cds_data)

    cds_filtered_df = pd.DataFrame(meta_cds_data)
    cds_filtered_df["main_category"] = "Digital Music"
    cds_filtered_df.to_csv("dataset/crossDomainData/filtered_data/meta/meta_CDs_and_Vinyl.csv", index=False)

def filter_games_meta_data(wanted_keys):
    '''
    Filter Video Games (Games) data through crossDomainData
    '''
    inter_games_df = pd.read_csv("dataset/crossDomainData/org_data/inter/Video_Games.csv", encoding="utf-8")
    games_parent_asin_set = set(inter_games_df["parent_asin"].unique())

    # Filter item IDs and export to CSV
    meta_games_data = []
    with open("dataset/crossDomainData/org_data/meta/meta_Video_Games.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            games_data = json.loads(line.strip())
            if games_data.get("parent_asin") in games_parent_asin_set and str(games_data.get("title")) != "nan" and games_data.get("title") and str(games_data.get("main_category")) != "nan" and games_data.get("main_category"):
                if games_data["categories"]:
                    games_data["categories"] = ";".join(games_data["categories"])
                games_data = {key: value for key, value in games_data.items() if key in wanted_keys}
                meta_games_data.append(games_data)

    games_filtered_df = pd.DataFrame(meta_games_data)
    games_filtered_df["main_category"] = "Video Games"
    games_filtered_df.to_csv("dataset/crossDomainData/filtered_data/meta/meta_Video_Games.csv", index=False)

def filter_meta_data():
    filter_cds_meta_data(wanted_keys={'parent_asin', 'title', 'main_category', "subtitle", "average_rating", "rating_number", "price", "store", "categories"})
    filter_games_meta_data(wanted_keys={'parent_asin', 'title', 'main_category', "subtitle", "average_rating", "rating_number", "price", "store", "categories"})
    filter_books_meta_data(wanted_keys={'parent_asin', 'title', 'main_category', "subtitle", "author", "average_rating", "rating_number", "price", "store", "categories"})
    filter_movies_meta_data(wanted_keys={'parent_asin', 'title', 'main_category', "subtitle", "average_rating", "rating_number", "price", "store", "categories"})
