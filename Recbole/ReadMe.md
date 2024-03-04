## How to start
recbole은 csv형태의 데이터를 사용하지 않는다. 
데이터셋을 atomic files로 따로 변경후 dataset폴더를 만들어 그 안에 atomic files 폴더를 넣어주세요.
dataset 폴더의 위치는 run.py와 일치하게 해주세요.
ex) dataset/level3/level3.inter

라이브러리를 많이 사용하기 떄문에 가상환경을 새로 만들것을 권장합니다.
1. conda create -n recbole
2. pip install -r requirements.txt
3. python run.py

## inference

소스코드를 보고 둘중에 하나를 선택해서 터미널에 입력하시면 됩니다.
사용하는 모델의 full_sort_pred 함수를 확인하고 리턴값에 view(-1)이 되어있다면 inference1을 사용
inference1은 submission.csv파일을 사용하는데 경로를 본인의 파일 경로에 맞춰서 수정해주세요.

python rum_inference1.py
python rum_inference2.py

## detail

1. 세부사항 조정은 recbole 폴더안에 있는 properties 폴더의 yaml파일을 수정해주세요
2. kmeans_pytorch가 필요하다고 뜰 수 있습니다. 그냥 pip install 해주세요