import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

train_df = pd.read_csv("/opt/ml/input/data/train/train_ratings.csv")  # 전체 학습 데이터
print("train shape: ", train_df.shape)

# side information data
year_data = pd.read_csv("/opt/ml/input/data/train/years.tsv", sep="\t")
writer_data = pd.read_csv("/opt/ml/input/data/train/writers.tsv", sep="\t")
title_data = pd.read_csv("/opt/ml/input/data/train/titles.tsv", sep="\t")
genre_data = pd.read_csv("/opt/ml/input/data/train/genres.tsv", sep="\t")
director_data = pd.read_csv("/opt/ml/input/data/train/directors.tsv", sep="\t")

test_df = pd.read_csv("/opt/ml/input/data/eval/sample_submission.csv")  # 예제 데이터
print("test shape: ", test_df.shape)


movie_train_df = pd.merge(train_df, title_data, on="item")
movie_train_df = movie_train_df.sort_values(["user", "time"]).reset_index(drop=True)

# 각 유저별로 영화 목록을 띄어쓰기로 구분하여 합치기
usb = movie_train_df.groupby("user")["title"].agg(" ".join).reset_index()
usb.nunique()


allvec = title_data["title"].tolist() + usb["title"].tolist()
tfidf = TfidfVectorizer()
moviematrix = tfidf.fit_transform(allvec).toarray()  # 각 TF-idf 를 계산합니다.
print("all tilte + user summary title shape: ", moviematrix.shape)

movie_vec, user_vec = np.split(moviematrix, [6807])
print("title vector shape: ", movie_vec.shape)

print("user summary title shape: ", user_vec.shape)

eu = euclidean_distances(user_vec, movie_vec)
co = cosine_similarity(user_vec, movie_vec)


# label encode user and item
user2idx = {entity: index for index, entity in enumerate(train_df["user"].unique())}
idx2user = {v: k for k, v in user2idx.items()}
item2idx = {entity: index for index, entity in enumerate(title_data["item"].unique())}
idx2item = {v: k for k, v in item2idx.items()}
train_df["user"] = train_df["user"].map(user2idx)
train_df["item"] = train_df["item"].map(item2idx)

# remove already seen items
user_indices = train_df["user"].values
item_indices = train_df["item"].values
co[user_indices, item_indices] = -1.0
eu[user_indices, item_indices] = 999

# train_df 초기화
data_path = "./data/train"
train_df = pd.read_csv("/opt/ml/input/data/train/train_ratings.csv")  # 전체 학습 데이터

eu_matrix = pd.DataFrame(
    eu, index=test_df["user"].unique(), columns=title_data["item"].unique()
)
co_matrix = pd.DataFrame(
    co, index=test_df["user"].unique(), columns=title_data["item"].unique()
)

# 폴더 이름 설정
folder_name = "output"

# 해당 폴더가 존재하지 않으면 생성
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"폴더 '{folder_name}'가 생성되었습니다.")
else:
    print(f"폴더 '{folder_name}'가 이미 존재합니다.")

# 각 행에서 가장 작은 값 10개가 위치한 열의 이름을 저장
smallest_columns_per_row = eu_matrix.apply(
    lambda row: row.nsmallest(10).index.tolist(), axis=1
)

# 모든 행의 결과를 하나의 리스트에 모으기
all_smallest_columns = smallest_columns_per_row.sum()
test_df["item"] = all_smallest_columns
test_df.to_csv("output/tf_df_eu.csv", index=False)

# 각 행에서 가장 작은 값 10개가 위치한 열의 이름을 저장
largest_columns_per_row = co_matrix.apply(
    lambda row: row.nlargest(10).index.tolist(), axis=1
)

# 모든 행의 결과를 하나의 리스트에 모으기
all_largest_columns = largest_columns_per_row.sum()
test_df["item"] = all_largest_columns
test_df.to_csv("output/tf_df_co.csv", index=False)
