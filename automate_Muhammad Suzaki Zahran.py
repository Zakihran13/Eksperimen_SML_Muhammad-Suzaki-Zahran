from pathlib import Path
import re
import shutil

import emoji
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from wordcloud import WordCloud


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR
RAW_DATA_PATH = PROJECT_DIR / "twitch_streams_raw_data.csv"
EDA_DIR = BASE_DIR / "eda outputs"
PREPROCESSED_DIR = BASE_DIR / "preprocessed data"


def load_data() -> pd.DataFrame:
	print("Loading dataset from Hugging Face...")
	dataset = load_dataset("ksang/TwitchStreams", split="train")
	df = dataset.to_pandas()
	print(f"Dataset loaded. Shape: {df.shape}")
	RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(RAW_DATA_PATH, index=False)
	print(f"Raw data saved to: {RAW_DATA_PATH}")
	return df


def extract_emojis(text: str) -> str:
	return "".join(char for char in str(text) if char in emoji.EMOJI_DATA)


def is_all_caps(text: str) -> int:
	alpha_text = re.sub(r"[^a-zA-Z]", "", str(text))
	if len(alpha_text) > 3 and alpha_text.isupper():
		return 1
	return 0


def add_eda_features(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df["char_count"] = df["text"].apply(len)
	df["word_count"] = df["text"].apply(lambda value: len(str(value).split()))
	df["emojis"] = df["text"].apply(extract_emojis)
	df["emoji_count"] = df["emojis"].apply(len)
	df["is_all_caps"] = df["text"].apply(is_all_caps)
	df["polarity"] = df["text"].apply(lambda value: TextBlob(str(value)).sentiment.polarity)
	df["subjectivity"] = df["text"].apply(lambda value: TextBlob(str(value)).sentiment.subjectivity)
	df["sentiment_category"] = pd.cut(
		df["polarity"],
		bins=[-1, 0, 1],
		labels=["SAFE", "MATURE"],
	)
	return df


def add_wordcloud_to_axis(ax, df: pd.DataFrame, content_label: str, title: str) -> None:
	text_data = " ".join(df[df["label"] == content_label]["text"].dropna().tolist())
	text_data = re.sub(r"http\S+|www\S+|https\S+", "", text_data, flags=re.MULTILINE)

	if text_data.strip():
		wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
		ax.imshow(wordcloud, interpolation="bilinear")
		ax.set_title(title, fontsize=14)
	else:
		ax.text(0.5, 0.5, f"No text data found for {content_label}", ha="center", va="center", fontsize=12)

	ax.axis("off")


def generate_eda_outputs(df: pd.DataFrame) -> None:
	if EDA_DIR.exists():
		shutil.rmtree(EDA_DIR)
	EDA_DIR.mkdir(parents=True, exist_ok=True)

	sns.set_theme(style="whitegrid")
	fig = plt.figure(figsize=(15, 16))
	ax_count = plt.subplot2grid((3, 2), (0, 0))
	ax_box = plt.subplot2grid((3, 2), (0, 1))
	ax_scatter = plt.subplot2grid((3, 2), (1, 0), colspan=2)
	ax_wc_safe = plt.subplot2grid((3, 2), (2, 0))
	ax_wc_mature = plt.subplot2grid((3, 2), (2, 1))

	sns.countplot(data=df, x="label", hue="label", palette="viridis", legend=False, ax=ax_count)
	ax_count.set_title("Distribution of Content Labels", fontsize=14)

	sns.boxplot(data=df, x="label", y="polarity", hue="label", palette="coolwarm", legend=False, ax=ax_box)
	ax_box.set_title("Sentiment Polarity Distribution by Content Label", fontsize=14)

	sns.scatterplot(data=df, x="word_count", y="polarity", alpha=0.5, hue="label", ax=ax_scatter)
	ax_scatter.set_title("Word Count vs Sentiment Polarity", fontsize=14)

	add_wordcloud_to_axis(ax_wc_safe, df, "SAFE", "Most Frequent Words in SAFE Streams")
	add_wordcloud_to_axis(ax_wc_mature, df, "MATURE", "Most Frequent Words in MATURE Streams")

	plt.tight_layout()
	dashboard_path = EDA_DIR / "eda_dashboard.png"
	fig.savefig(dashboard_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"EDA dashboard saved to: {dashboard_path}")

	summary_stats = df.groupby("label")[["char_count", "word_count", "emoji_count", "polarity"]].mean()
	summary_path = EDA_DIR / "summary_statistics.csv"
	summary_stats.to_csv(summary_path)
	print(f"Summary statistics saved to: {summary_path}")

	negative_path = EDA_DIR / "top_5_most_negative.csv"
	positive_path = EDA_DIR / "top_5_most_positive.csv"
	df.sort_values(by="polarity").head(5)[["text", "polarity", "label"]].to_csv(negative_path, index=False)
	df.sort_values(by="polarity", ascending=False).head(5)[["text", "polarity", "label"]].to_csv(positive_path, index=False)
	print(f"Top negative examples saved to: {negative_path}")
	print(f"Top positive examples saved to: {positive_path}")


def clean_text(text: str) -> str:
	text = str(text).lower()
	text = re.sub(r"http\S+|www\.\S+", " ", text)
	text = re.sub(r"@\w+", " ", text)
	text = re.sub(r"#(\w+)", r"\1", text)
	text = re.sub(r"[^a-z\s]", " ", text)
	text = re.sub(r"\s+", " ", text).strip()
	tokens = [token for token in text.split() if token not in ENGLISH_STOP_WORDS and len(token) > 1]
	return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str, str | None]:
	df_prep = df.copy()
	print("=== PREPROCESSING CHECKLIST ===")
	print(f"Ukuran awal data: {df_prep.shape}")

	text_col_candidates = ["text", "content", "caption", "tweet", "comment"]
	label_col_candidates = ["label", "target", "class", "sentiment"]
	text_col = next((column for column in text_col_candidates if column in df_prep.columns), None)
	label_col = next((column for column in label_col_candidates if column in df_prep.columns), None)

	if text_col is None:
		raise ValueError("Kolom teks tidak ditemukan. Pastikan ada kolom seperti: text/content/caption/tweet/comment")

	print(f"Kolom teks terdeteksi: {text_col}")
	print(f"Kolom label terdeteksi: {label_col if label_col else 'Tidak ada (opsional)'}")

	missing_before = df_prep.isna().sum().sum()
	print(f"\nTotal missing values sebelum cleaning: {missing_before}")

	df_prep = df_prep.dropna(subset=[text_col]).copy()
	if label_col:
		df_prep = df_prep.dropna(subset=[label_col]).copy()

	numeric_cols = df_prep.select_dtypes(include=[np.number]).columns
	for column in numeric_cols:
		if df_prep[column].isna().sum() > 0:
			df_prep[column] = df_prep[column].fillna(df_prep[column].median())

	object_cols = df_prep.select_dtypes(exclude=[np.number]).columns
	for column in object_cols:
		if df_prep[column].isna().sum() > 0:
			mode_value = df_prep[column].mode(dropna=True)
			fill_value = mode_value.iloc[0] if not mode_value.empty else "unknown"
			df_prep[column] = df_prep[column].fillna(fill_value)

	missing_after = df_prep.isna().sum().sum()
	print(f"Total missing values sesudah cleaning: {missing_after}")

	duplicate_rows = df_prep.duplicated().sum()
	duplicate_text = df_prep.duplicated(subset=[text_col]).sum()
	df_prep = df_prep.drop_duplicates().copy()
	df_prep = df_prep.drop_duplicates(subset=[text_col]).copy()

	print(f"\nDuplikat full-row yang dihapus: {duplicate_rows}")
	print(f"Duplikat teks yang dihapus: {duplicate_text}")

	df_prep["clean_text"] = df_prep[text_col].apply(clean_text)
	df_prep = df_prep[df_prep["clean_text"].str.len() > 0].copy()
	df_prep["clean_char_count"] = df_prep["clean_text"].str.len()
	df_prep["clean_word_count"] = df_prep["clean_text"].str.split().str.len()

	if label_col:
		model_df = df_prep[["clean_text", label_col]].copy()
	else:
		model_df = df_prep[["clean_text"]].copy()

	print("\n=== RINGKASAN HASIL PREPROCESSING ===")
	print(f"Ukuran akhir data: {df_prep.shape}")
	print(model_df.head())
	return df_prep, model_df, text_col, label_col


def export_split(model_df: pd.DataFrame, label_col: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
	if PREPROCESSED_DIR.exists():
		shutil.rmtree(PREPROCESSED_DIR)
	PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

	stratify_target = model_df[label_col] if label_col and label_col in model_df.columns else None
	train_df, test_df = train_test_split(
		model_df,
		test_size=0.2,
		random_state=42,
		stratify=stratify_target,
	)

	train_df = train_df.reset_index(drop=True)
	test_df = test_df.reset_index(drop=True)

	train_path = PREPROCESSED_DIR / "train.csv"
	test_path = PREPROCESSED_DIR / "test.csv"
	train_df.to_csv(train_path, index=False)
	test_df.to_csv(test_path, index=False)

	print("Train-test split berhasil dibuat dan disimpan.")
	print(f"Train shape: {train_df.shape}")
	print(f"Test shape: {test_df.shape}")
	print(f"Train file: {train_path}")
	print(f"Test file: {test_path}")
	return train_df, test_df


def main() -> None:
	df = load_data()
	df = add_eda_features(df)
	generate_eda_outputs(df)
	df_prep, model_df, _, label_col = preprocess_dataframe(df)
	preprocessed_full_path = PREPROCESSED_DIR / "preprocessed_full.csv"
	if PREPROCESSED_DIR.exists():
		shutil.rmtree(PREPROCESSED_DIR)
	PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
	df_prep.to_csv(preprocessed_full_path, index=False)
	print(f"Full preprocessed data saved to: {preprocessed_full_path}")
	export_split(model_df, label_col)


if __name__ == "__main__":
	main()
