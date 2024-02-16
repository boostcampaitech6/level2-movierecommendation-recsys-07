# run_inference.py 구현 예시

import argparse
import torch
import numpy as np
import pandas as pd
import os

from recbole.quick_start import load_data_and_model

# BPR, NGCF
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, default='saved/BPR-Feb-07-2024_05-34-57.pth', help='name of models')
    # python run_inference.py --model_path=saved/BPR-Feb-07-2024_05-34-57.pth 로 실행
    
    args, _ = parser.parse_known_args()
    
    # model, dataset 불러오기
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(args.model_path)
    
    # device 설정
    device = config.final_config_dict['device']
    
    # user, item id -> token 변환 array
    user_id2token = dataset.field2id_token['user_id']
    item_id2token = dataset.field2id_token['item_id']
    
    # user-item sparse matrix
    matrix = dataset.inter_matrix(form='csr')
    # user id, predict item id 저장 변수
    pred_list = None
    user_list = None
    model.eval()
    for data in test_data:
        interaction = data[0].to(device)
        score = model.full_sort_predict(interaction)
        rating_pred = score.cpu().data.numpy().copy()
        batch_user_index = interaction['user_id'].cpu().numpy()
        rating_pred[matrix[batch_user_index].toarray() > 0] = 0
        ind = np.argpartition(rating_pred, -10)[:, -10:]
        
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

        batch_pred_list = ind[
            np.arange(len(rating_pred))[:, None], arr_ind_argsort
        ]
        
        # 예측값 저장
        if pred_list is None:
            pred_list = batch_pred_list
            user_list = batch_user_index
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            user_list = np.append(user_list, batch_user_index, axis=0)
        
    result = []
    for user, pred in zip(user_list, pred_list):
        for item in pred:
            result.append((int(user_id2token[user]), int(item_id2token[item])))

    # 폴더 이름 설정
    folder_name = 'output'

    # 폴더가 있는지 확인하고, 없으면 만들기
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)       
    
    # 데이터 저장
    dataframe = pd.DataFrame(result, columns=["user", "item"])
    dataframe.sort_values(by='user', inplace=True)
    dataframe.to_csv(
        folder_name+"/submission.csv", index=False
    )
    
    print('inference done!')