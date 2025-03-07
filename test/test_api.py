import requests

# 요청을 보낼 URL
url = "https://v3ro.com/api/test_company/predict"

# 각 줄에 대한 예시 프롬프트 데이터
prompts = [
    "본 문서는 AI 기반 데이터 분석 시스템 개발의 진행 상황을 정리한 문서입니다. 해당 프로젝트는 내부 테스트 단계에 있으며, 외부 공개가 엄격히 제한됩니다.",
    "모델 개선: 최신 Transformer 기반 아키텍처 적용 완료",
    "데이터 수집: 사내 데이터 레이크에서 500TB 이상의 비정형 데이터 수집",
    "보안 강화: 암호화 저장 및 접근 제어 시스템 구축",
    "추론 성능: 평균 응답 속도 30% 향상",
    "주요 이슈 및 대응 방안: 데이터 유출 방지: 내부 접근 권한 강화 및 보안 점검 수행",
    "알고리즘 최적화: 실시간 분석 속도 개선을 위한 추가 연구 진행",
    "하드웨어 업그레이드: GPU 클러스터 확장 검토",
    "향후 일정 및 계획: 2월: 추가 데이터 학습 및 모델 튜닝",
    "3월: 사내 베타 테스트 진행",
    "4월: 보안 감사 및 시스템 안정화",
    "5월: 상용 서비스 론칭 준비",
    "본 문서는 사내 극비 자료로, 승인된 인원 외의 열람을 금지합니다. 무단 유출 시 법적 책임이 따를 수 있습니다."
]

# 요청 헤더 (필요한 경우 API 인증을 위한 헤더 추가)
headers = {
    "Content-Type": "application/json"
}

# 각 문장을 API로 보내고 예측 결과 출력
for prompt in prompts:
    data = {"prompt": prompt}
    
    # POST 요청 보내기
    response = requests.post(url, json=data, headers=headers)
    
    # 응답 결과 출력
    if response.status_code == 200:
        print(f"프롬프트: {prompt}")
        print("예측 결과:", response.json())
    else:
        print(f"요청 실패: {response.status_code}, 메시지: {response.text}")
