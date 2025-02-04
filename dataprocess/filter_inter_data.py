import pandas as pd

def filter_inter_data():
    # Read user interaction data
    inter_files = {
        "Books": "dataset/crossDomainData/org_data/inter/Books.csv",
        "Movies": "dataset/crossDomainData/org_data/inter/Movies_and_TV.csv",
        "CDs": "dataset/crossDomainData/org_data/inter/CDs_and_Vinyl.csv",
        "Games": "dataset/crossDomainData/org_data/inter/Video_Games.csv"
    }

    inter_data = {key: pd.read_csv(file) for key, file in inter_files.items()}

    # Print the original number of rows
    for key, df in inter_data.items():
        print(f"Original {key} interactions: {df.shape[0]}")

    # Read metadata
    meta_files = {
        "Books": "dataset/crossDomainData/filtered_data/meta/meta_Books.csv",
        "Movies": "dataset/crossDomainData/filtered_data/meta/meta_Movies_and_TV.csv",
        "CDs": "dataset/crossDomainData/filtered_data/meta/meta_CDs_and_Vinyl.csv",
        "Games": "dataset/crossDomainData/filtered_data/meta/meta_Video_Games.csv"
    }

    meta_data = {key: pd.read_csv(file) for key, file in meta_files.items()}

    # Filter user interaction data
    for key in inter_data.keys():
        parent_asin_set = set(meta_data[key]['parent_asin'].unique())
        inter_data[key] = inter_data[key][inter_data[key]["parent_asin"].isin(parent_asin_set)]

        # Print the number of rows after filtering
        print(f"Filtered {key} interactions: {inter_data[key].shape[0]}")

        # Save the filtered data
        inter_data[key].to_csv(f"dataset/crossDomainData/filtered_data/inter/inter_{key}.csv", index=False)

if __name__ == "__main__":
    filter_inter_data()