import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os
import re
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

# preprocess #####################################################################################

# 1900년부터 2099년까지의 년도 추출 및 새로운 컬럼 생성
title_data["year"] = title_data["title"].str.extract(r"\((19[0-9]{2}|20[0-9]{2})\)")

# 예외 데이터에 대한 처리
# "Fawlty Towers (1975-1979)"의 경우 1975 추출, "Big Bang Theory, The (2007-)"의 경우 2007 추출
title_data["year"] = title_data.apply(
    lambda row: (
        re.search(r"\((\d{4})", row["title"]).group(1)
        if pd.isna(row["year"])
        else row["year"]
    ),
    axis=1,
)

# 'title' 컬럼에서 괄호와 괄호 안의 내용 삭제
title_data["title"] = (
    title_data["title"].str.replace(r"\(.*?\)", "", regex=True).str.strip()
)

# 관사 및 전치사를 포함하는 정규 표현식 패턴
pattern = r"\b(?:a|an|the|for|in|of|on|to|with)\b"

# 관사와 전치사 제거
title_data["title"] = (
    title_data["title"].str.replace(pattern, "", flags=re.IGNORECASE).str.strip()
)

# 'year' 컬럼을 문자열로 변환하고 'title' 컬럼과 결합
title_data["title"] = title_data["title"] + " " + title_data["year"].astype(str)
title_data = title_data.drop("year", axis=1)

# director #######################################################################################

title_data = pd.merge(title_data, director_data, on="item", how="outer")
# 'director' 컬럼에서 NaN 값을 빈 문자열로 변환하고 결합
title_data = (
    title_data.groupby(["item", "title"])["director"]
    .agg(lambda x: " ".join(x.fillna("")))
    .reset_index()
)
# 'director' 컬럼을 'title' 컬럼과 결합
title_data["title"] = title_data["title"] + " " + title_data["director"]
title_data = title_data.drop("director", axis=1)

# writer #########################################################################################

title_data = pd.merge(title_data, writer_data, on="item", how="outer")
# 'writer' 컬럼에서 NaN 값을 빈 문자열로 변환하고 결합
title_data = (
    title_data.groupby(["item", "title"])["writer"]
    .agg(lambda x: " ".join(x.fillna("")))
    .reset_index()
)
# 'writer' 컬럼을 'title' 컬럼과 결합
title_data["title"] = title_data["title"] + " " + title_data["writer"]
title_data = title_data.drop("writer", axis=1)

# genre ##########################################################################################

title_data = pd.merge(title_data, genre_data, on="item", how="outer")
# 'genre' 컬럼에서 NaN 값을 빈 문자열로 변환하고 결합
title_data = (
    title_data.groupby(["item", "title"])["genre"]
    .agg(lambda x: " ".join(x.fillna("")))
    .reset_index()
)
# 'genre' 컬럼을 'title' 컬럼과 결합
title_data["title"] = title_data["title"] + " " + title_data["genre"]
title_data = title_data.drop("genre", axis=1)

##################################################################################################


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
