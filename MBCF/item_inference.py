import numpy as np
import pandas as pd
from tqdm import tqdm

# load csv file to df
df = pd.read_csv("/opt/ml/input/data/train/train_ratings.csv")

# get number of users, number of items
n_users = df["user"].nunique()
n_items = df["item"].nunique()
print(n_users, n_items)

# label encode user and item
user2idx = {entity: index for index, entity in enumerate(df["user"].unique())}
idx2user = {v: k for k, v in user2idx.items()}
item2idx = {entity: index for index, entity in enumerate(df["item"].unique())}
idx2item = {v: k for k, v in item2idx.items()}
df["user"] = df["user"].map(user2idx)
df["item"] = df["item"].map(item2idx)

# df to sparse matrix
train_matrix = np.zeros((n_users, n_items), dtype=float)

user_indices = df["user"].values
item_indices = df["item"].values
train_matrix[user_indices, item_indices] = 1.0


try:
    result_matrix = np.load("./item_result_matrix.npy")
    print("npy loaded")
except:
    # calculate cosine similarity
    print("similarity")
    item_sim = np.ones((n_items, n_items), dtype=float)
    """
    for i in tqdm(range(n_items)):
        item_sim[i, i] = 1.0
        for j in range(i+1, n_items):
            cosine_sim = np.inner(train_matrix[:,i],train_matrix[:,j]) / (np.sum(train_matrix[:,i]) * np.sum(train_matrix[:,j]))
            item_sim[i, j] = cosine_sim
            item_sim[j, i] = cosine_sim
    """
    train_matrix_normalized = train_matrix / np.linalg.norm(
        train_matrix, axis=0, keepdims=True
    )
    item_sim = train_matrix_normalized.T @ train_matrix_normalized
    item_sim[np.isnan(item_sim)] = 0.0
    np.fill_diagonal(item_sim, 1.0)

    # perform collaborative filetering
    print("CF")
    """result_matrix = np.zeros((n_users, n_items), dtype=float)
    for i in tqdm(range(n_items)):
        for j in range(n_items):
            result_matrix[:,i] += train_matrix[:,j] * item_sim[i, j]
    result_matrix /= np.sum(item_sim, axis=0)"""
    result_matrix = train_matrix @ item_sim
    np.save("./item_result_matrix.npy", result_matrix)
print("cf_matrix shape:", result_matrix.shape)


# remove already seen items
user_indices = df["user"].values
item_indices = df["item"].values
result_matrix[user_indices, item_indices] = -1.0

# get top 10 item for each user
print("Rank")
top_n = 10
top_indices = np.argsort(result_matrix, axis=1)[:, -top_n:]

# make submission file
submission_df = pd.DataFrame(columns=["user", "item"])

for i in range(n_users):
    user_id = idx2user[i]
    top_items = [idx2item[idx] for idx in top_indices[i]]
    submission_df = pd.concat(
        (submission_df, pd.DataFrame({"user": [user_id] * top_n, "item": top_items}))
    )
submission_df.to_csv("item_submission.csv", index=False)
