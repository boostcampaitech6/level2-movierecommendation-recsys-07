# run_inference.py

import argparse
import torch
import numpy as np
import pandas as pd

from recbole.quick_start import load_data_and_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, default='saved/NGCF-Feb-07-2024_13-16-54.pth', help='name of models')
    
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
        user_id = interaction['user_id'].cpu().numpy()
        
        # 사용자가 상호작용한 아이템 인덱스를 가져옵니다.
        interacted_indices = matrix[user_id].indices

        # 상호작용한 아이템의 점수를 0으로 설정합니다.
        rating_pred[interacted_indices] = 0

        ind = np.argpartition(rating_pred, -10)[-10:]
        
        arr_ind = rating_pred[ind]
       
       # 추출된 값들을 내림차순으로 정렬하기 위한 인덱스를 얻음
        arr_ind_argsort = np.argsort(arr_ind)[::-1]

        # 실제 값들을 정렬된 순서대로 인덱스 배열에 적용
        batch_pred_list = ind[arr_ind_argsort]
        
        # 예측값 저장
        if pred_list is None:
            pred_list = batch_pred_list
            # batch_pred_list 길이만큼 user_id를 반복
            user_list = np.repeat(user_id, len(batch_pred_list))
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            # batch_pred_list 길이만큼 user_id를 반복하여 추가
            user_list = np.append(user_list, np.repeat(user_id, len(batch_pred_list)), axis=0)
    
    test_df = pd.read_csv('data/sample_submission.csv') # 예제 데이터
    print('test shape: ', test_df.shape)

    result = []
    for item in pred_list:
        result.append(int(item_id2token[item]))
    
    test_df['item'] = result
    test_df.to_csv('output/submission.csv', index=False)
    print('Final mapping done and saved to CSV!')