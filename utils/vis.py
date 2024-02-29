import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import json
import os
import glob




def get_result_df(result_path, wchich_parser = "score3", context_length = 32000):
    folder_path = result_path
    # Path to the directory containing JSON results
    # Using glob to find all json files in the directory
    json_files = glob.glob(f"{folder_path}/*.json")

    # List to hold the data
    data = []

    # Iterating through each file and extract the 3 columns we need
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            # Extracting the required fields
            document_depth = json_data.get("depth_percent", None)
            context_length = json_data.get("context_length", None)
            score = json_data.get(wchich_parser, None)
            # Appending to the list
            data.append({
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score
            })

    # Creating a DataFrame
    df = pd.DataFrame(data)
    df = df[df["Context Length"] <= context_length]
    print (df.head())
    print (f"You have {len(df)} rows")
    return df


def get_pivot_table(df):
    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    pivot_table.iloc[:5, :5]
    return pivot_table


def get_figure(pivot_table, title, save_path):
    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        cmap=cmap,
        cbar_kws={'label': 'Score'}
    )

    # More aesthetics
    # plt.title('Pressure Testing GPT-4 128K Context\nFact Retrieval Across Context Lengths ("Needle In A HayStack")')  # Adds a title
    plt.title(title)  # Adds a title
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    plt.savefig(save_path, dpi=300)
    # Show the plot
    plt.show()

def result_analysis(result_path, model_variant, inject, mode, save_path, figure_suffix, wchich_parser = "score3", context_length = 32000):
    title = f"[{wchich_parser}_{mode}]{inject} {model_variant} {figure_suffix}"
    df = get_result_df(result_path, wchich_parser, context_length)
    pivot_table = get_pivot_table(df)
    get_figure(pivot_table, title, save_path)
    return pivot_table