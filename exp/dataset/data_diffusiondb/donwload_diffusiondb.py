from urllib.request import urlretrieve


# Download the parquet table
table_url = f"https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet"
urlretrieve(table_url, "metadata.parquet")

# # Read the table using Pandas
# import pandas as pd
# metadata_df = pd.read_parquet('metadata.parquet')
