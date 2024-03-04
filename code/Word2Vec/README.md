# 코드 흐름
python word2vec.py
python visualize.py
python inertia.py
python clustering.py

# 코드 역할
## word2vec.py
Raw data 읽기 -> user 단위로 sentence 생성 -> Word2Vec 모델 학습 및 저장 -> bset 모델 Load ->
모델로 학습한 embedding csv 저장 -> TSNE로 2차원 축소 후 csv 저장 -> 장르별로 시각화

## visualize.py
TSNE csv 파일 읽기 -> year별로 시각화

## inertia.py
inertia: KMeans Clustering을 수행했을 때 해당 cluster가 얼마나 잘 모여있는지 나타내는 수치
k가 늘어남에 따라 inertia는 감소하며, 이 감소폭이 급격히 완만해지는 지점의 k가 적절한 cluster 수

## clustering.py
TSNE csv 파일 읽기 -> cluster별로 시각화
cluster의 수는 inertia.py의 결과 그래프를 보고 판단함


# 참고 사항
- 결과물 중 not_shuffle이 이름에 들어간 것은 Word2Vec 모델을 학습할 때 유저의 sentence를 섞지 않은 버전의 결과물임
- 코드에 파일명이 하드코딩 되어 있는 경우가 많아 코드 흐름을 숙지 후 적절히 변경해서 사용해야 함