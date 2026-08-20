# 한국형 실리콘 샘플링 검증

LLM에 인구통계 페르소나를 프롬프트로 부여해 한국 여론조사 문항(KGSS)에
응답시키고, 가상 응답이 실제 응답을 얼마나 재현하는지 정량 분석한다.

## 데이터

원본 파일은 저장소에 포함되지 않는다.
KOSSDA(kossda.snu.ac.kr)에서 개별 신청 후 프로젝트 루트에 둘 것.

- `2003-2025_KGSS_kor_public.sav`
- `2003-2025_KGSS_codebook.pdf`

## 환경 구성

```powershell
uv venv --python 3.12
uv pip install -r requirements.txt
```

## 실행 순서

```powershell
# 1. 데이터 구조 진단 (결측 4층 분리, eta2, 셀 크기)
uv run kgss_inventory.py --sav "2003-2025_KGSS_kor_public.sav" `
    --out .\inventory --target-year 2023

# 2. 코드북에서 설문 원문 추출 + A/B 분할표본 분석
uv run kgss_codebook.py --pdf "2003-2025_KGSS_codebook.pdf" `
    --inv .\inventory --out .\selection

# 3. (선택) 사람이 읽는 코드북/회차 데이터 변환
uv run kgss_export.py --sav "2003-2025_KGSS_kor_public.sav" --year 2023

## 주요 산출물

| 파일 | 내용 |
|---|---|
| `inventory/kgss_clean.parquet` | 결측 정리 완료 데이터. 이후 모든 분석의 입력 |
| `inventory/diagnostics.md` | 데이터 진단 요약 |
| `selection/variables_worded.csv` | 전 변수 설문 원문 + 선택지. 프롬프트 원천 |
| `selection/ab_pairs.csv` | A/B 분할표본과 인간 응답 순서 효과 |

## 데이터 구조 메모

- 결측은 4층: `-1`(회차 미조사, 전체 셀의 89.6%) / `-8`(형식 선언) /
  `IAP`·`9`·`99`(비해당) / `8`·`88`(모름·무응답)
- KGSS는 패널이 아닌 반복 횡단면. 개인 수준 분석은 단일 회차 내에서만 가능
- `FINALWT`는 회차 내 정규화 가중치. 연도 통합 가중 분석 불가
- 층화·집락 변수 미공개 → 설계효과 반영 불가, 신뢰구간이 실제보다 좁음
