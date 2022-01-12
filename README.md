# KoBERT-multi-classification-model
- 감정 다중 분류 KoBERT 모델
- 목적: KoBERT 다중분류 모델을 이용한 감정분석 기반 영화 추천 시스템 연구(A study on the Movie Recommendation System based on Emotional Analysis using the KoBERT Multi-classification Model)
- 연구 진행 날짜: 2021년 9~12월(약 3개월)

## 1. 개요
- 기존의 감성 분석은 평가의 긍/부정 극성을 판별하거나 특징별 평가 요약 등에만 초점을 맞추고 있어 상대적으로 의견 정보의 활용도가 낮아지는 문제가 있다(연종흠 외 3인, 2011). 해당 연구는 긍/부정을 나타내는 감성(Sentimental) 수준에서의 분류가 아닌 소비자의 감정(Emotional)을 고려해 리뷰 데이터에 나타난 소비자의 반응이 어떤 감정을 나타내는지에 대해 분석한 결과가 다른 소비자에게 영향을 미칠 수 있을 것이라고 판단하여 연구를 진행하였다.
- KoBERT 다중분류 모델을 사용하여 네이버 영화 37개에 대한 총 358,199 문장을 7가지 감정(기쁨, 슬픔, 공포, 당황, 분노, 놀람, 중립)으로 분류
- 타겟값: 7가지 감정(기쁨, 슬픔, 놀람, 분노, 공포, 혐오, 중립)

## 2. 사용한 데이터셋(*출처: 한국지능정보사회진흥원)

A. 한국어 감정 정보가 포함된 단발성 대화 데이터셋
- 문장 갯수(개): 74,856
- 타겟값: 기쁨, 슬픔, 놀람, 분노, 공포, 혐오, 중립(7가지 감정)

B. 감성대화말뭉치
- 문장 갯수(개): 38,594
- 타겟값: 기쁨, 불안, 당황, 분노, 상처, 슬픔(6가지 감정)

C. 최종 KoBERT 모델에서 사용한 사전: A+B 데이터셋
- 문장 갯수(개): 113,450개
- 타겟값: 3번 모델 설계 이후 결정

## 3. KoBERT 모델 설계
1단계. A 데이터셋을 이용한 KoBERT 다중분류모델의 정확도 도출

2단계. B 데이터셋을 이용한 KoBERT 다중분류모델의 정확도 도출

3단계. 1단계와 2단계 모델의 정확도 비교를 통해, 더 높은 정확도를 갖고 있는 모델 확인
*더 높은 정확도를 갖고 있는 모델의 감정(타겟값)을 기준으로 다중 분류 KoBERT 모델 구현

4단계. B 데이터셋을 A 데이터셋의 7가지 감정 라벨로 재분류(re-labeling) 진행

5단계. A와 B 데이터셋을 합친 총 113,450개 문장으로 KoBERT 다중분류 모델 구현

(*세부 결과 및 이후 추천시스템 내용 생략)

**참고**
1. https://velog.io/@seolini43/KOBERT%EB%A1%9C-%EB%8B%A4%EC%A4%91-%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8-%EB%A7%8C%EB%93%A4%EA%B8%B0-%ED%8C%8C%EC%9D%B4%EC%8D%ACColab
2. https://github.com/SKTBrain/KoBERT/tree/master/kobert_hf
3. https://github.com/SKTBrain/KoBERT/issues/68
