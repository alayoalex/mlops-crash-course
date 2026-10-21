from pathlib import Path
import pandas as pd
from utils import create_balance_dataset, random_split, download_and_unzip_spam_data

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

root_path = Path(__file__).parent / "data" / "SPAM"
zip_path = root_path / "sms_spam_collection.zip"
extracted_path = root_path / "sms_spam_collection"
data_file_path = extracted_path / "SMSSpamCollection.tsv"

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
balance_df = create_balance_dataset(df)
balance_df["Label"] = balance_df["Label"].map({"ham": 0, "spam": 1})

train_df, validation_df, test_df = random_split(balance_df, 0.7, 0.1)
train_df.to_csv(root_path / "train.csv", index=None)
validation_df.to_csv(root_path / "validation.csv", index=None)
test_df.to_csv(root_path / "test.csv", index=None)
