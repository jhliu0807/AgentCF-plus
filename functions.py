import os

# Concatenate all user cross-domain memories
def concatenate_crossdomain_preference(folder_path):
    combined_content = ""
    # Traverse all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is in txt format and not named 1.txt
        if filename.endswith('.txt') and filename.startswith('private'):
            file_path = os.path.join(folder_path, filename)
            # Read the file content and concatenate
            with open(file_path, 'r', encoding='utf-8') as file:
                combined_content += f"--- preferences in {os.path.splitext(filename)[0].split('-')[-1]} ---\n"
                combined_content += file.read() + "\n\n"  # Add a newline to separate content

    return combined_content