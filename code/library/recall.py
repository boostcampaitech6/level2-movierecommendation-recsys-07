import pandas as pd
import argparse


def recall_at_10(submission_df: pd.DataFrame, valid_df: pd.DataFrame):
    sum_recall = 0
    true_users = 0

    for user in submission_df["user"].unique():
        pred_set = set(submission_df[submission_df["user"] == user]["item"])
        # assert len(pred_set) == 10, f"Length of Prediction is not 10 for user {user}"
        valid_set = set(valid_df[valid_df["user"] == user]["item"])

        if len(valid_set) > 0:
            sum_recall += len(pred_set & valid_set) / min(len(valid_set), 5)
            true_users += 1

    return sum_recall / true_users


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sub", type=str)
    parser.add_argument(
        "--valid", default="../data/train/custom_valid_ratings.csv", type=str
    )

    args = parser.parse_args()

    submission_df = pd.read_csv(args.sub)
    valid_df = pd.read_csv(args.valid)
    recall = recall_at_10(submission_df, valid_df)

    print(f"Recall@10: {recall}")
