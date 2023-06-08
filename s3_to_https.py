import pandas as pd

df = pd.read_csv(filepath_or_buffer="results_csv/results_fruits.csv", sep=",")

prefix_html = "https://ds-p8.s3.eu-west-1.amazonaws.com/data/"
prefix_s3 = "s3a://ds-p8/data/"

df.rename(mapper={"path": "url"}, axis=1, inplace=True)

df["url"] = df["url"].apply(lambda url: url.replace(prefix_s3, prefix_html))
