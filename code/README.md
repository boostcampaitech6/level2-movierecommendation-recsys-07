## How to run

0. Setting
   ```
   cd ..
   pip install -r requirements.txt
   python split_valid.py
   ```
1. Training
   ```
   python train.py [args]=[value]
   ```
2. Inference
   ```
   python inference.py -cp [saved_model_dir]
   ```

## Recall@10
```
python recall.py --sub [submission_file_path]
```

+ #### custom_train_ratings.csv로 학습한 결과 기준입니다. 기존 파일로 학습시 recall은 0이 됩니다.