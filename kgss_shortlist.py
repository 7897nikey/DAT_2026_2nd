r"""
문항 선정 작업지 생성

305개 후보를 3인이 독립 태깅할 수 있는 xlsx로 정리한다.
자동으로 채우는 정보와 사람이 판단할 열을 분리한다.

자동 채움:
    변수명 / 문항내용 / 선택지 / 응답수 / eta2 / eta2_구간 / 배터리 / AB분할표본
사람 판단 (빈 열):
    유형        사실 / 태도 / 민감
    포함        Y / N
    메모

배터리 탐지: 변수명 끝의 숫자를 떼어낸 어간이 같으면 같은 문항군으로 본다.
    CESD1..CESD11 -> CESD,  PERTRT1..PERTRT17 -> PERTRT
    같은 배터리에서 여러 문항을 뽑으면 조건-타깃 누출이 생기므로 표시해 둔다.

AB 분할표본 탐지: 어간이 같고 접미가 A/B인 쌍.
    SAMPTHOUA / SAMPTHOUB23,  ELEFRAUDA / ELEFRAUDB
    문구 효과의 인간 기준값이 데이터에 이미 있으므로 별도 축으로 쓴다.

사용법:
    uv pip install openpyxl
    uv run kgss_shortlist.py --inv .\inventory --out .\selection --year 2023
"""

import argparse
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

STRUCTURAL = {-8.0, -1.0}
TRAILING_NUM = re.compile(r"\d+$")
AB_SUFFIX = re.compile(r"^(?P<stem>.+?)(?P<ab>[AB])(?P<yr>\d{2})?$")


def battery_stem(var: str) -> str:
    """CESD11 -> CESD, GOVSPD3 -> GOVSPD, POSGEN10 -> POSGEN"""
    s = TRAILING_NUM.sub("", var)
    return s if len(s) >= 3 else var


def fmt_options(vl_json: str, dk_json: str, inap_json: str) -> str:
    try:
        vl = json.loads(vl_json)
    except Exception:
        return ""
    dk = set(json.loads(dk_json or "[]"))
    inap = set(json.loads(inap_json or "[]"))
    parts = []
    for k, lab in sorted(vl.items(), key=lambda kv: float(kv[0])):
        code = float(k)
        if code in STRUCTURAL or code in dk or code in inap:
            continue
        parts.append(f"{int(code) if code == int(code) else code}={lab}")
    return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inv", default="./inventory")
    ap.add_argument("--out", default="./selection")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--taggers", nargs="+", default=["주희", "태우", "다교"])
    args = ap.parse_args()

    inv, outdir = Path(args.inv), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    wave = pd.read_csv(inv / "wave_items.csv")
    vt = pd.read_csv(inv / "variables.csv")
    eta_path = inv / "eta2.csv"
    eta = pd.read_csv(eta_path) if eta_path.exists() else pd.DataFrame(columns=["item", "eta2"])

    cand = wave[wave["is_item_cand"]].copy()
    print(f"후보 {len(cand)}개")

    info = vt.set_index("var")
    cand["선택지"] = [
        fmt_options(
            info.at[v, "value_labels"] if v in info.index else "{}",
            info.at[v, "dk"] if v in info.index else "[]",
            info.at[v, "inap"] if v in info.index else "[]",
        )
        for v in cand["var"]
    ]

    cand = cand.merge(eta[["item", "eta2"]], left_on="var", right_on="item", how="left")
    cand.drop(columns=["item"], inplace=True, errors="ignore")

    # eta2 3분위
    q = cand["eta2"].dropna()
    if len(q) >= 10:
        lo, hi = q.quantile([1 / 3, 2 / 3])
        cand["eta2_구간"] = pd.cut(
            cand["eta2"], [-1, lo, hi, 2], labels=["저", "중", "고"]
        )
    else:
        cand["eta2_구간"] = np.nan

    # 배터리
    cand["배터리"] = cand["var"].map(battery_stem)
    size = cand["배터리"].value_counts()
    cand["배터리크기"] = cand["배터리"].map(size)

    # AB 분할표본 쌍
    stems = {}
    for v in cand["var"]:
        m = AB_SUFFIX.match(v)
        if m:
            stems.setdefault(m.group("stem"), []).append((m.group("ab"), v))
    ab_pairs = {}
    for stem, lst in stems.items():
        sides = {ab for ab, _ in lst}
        if {"A", "B"} <= sides:
            for _, v in lst:
                ab_pairs[v] = stem
    cand["AB분할표본"] = cand["var"].map(ab_pairs).notna().map({True: "Y", False: ""})
    n_ab = int((cand["AB분할표본"] == "Y").sum())
    print(f"AB 분할표본 문항 {n_ab}개")

    ycol = f"y{args.year}"
    sheet = cand[
        ["var", "label", "선택지", "n_options", ycol, "eta2", "eta2_구간",
         "배터리", "배터리크기", "AB분할표본", "ordinal_guess"]
    ].rename(
        columns={
            "var": "변수명",
            "label": "문항내용",
            "n_options": "선택지수",
            ycol: "응답률",
            "ordinal_guess": "순서형",
        }
    )
    sheet["응답률"] = sheet["응답률"].round(3)
    sheet["유형"] = ""   # 사실 / 태도 / 민감
    sheet["포함"] = ""   # Y / N
    sheet["메모"] = ""
    sheet = sheet.sort_values(["배터리크기", "배터리", "변수명"], ascending=[False, True, True])

    xlsx = outdir / f"문항선정_작업지_{args.year}.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        for name in args.taggers:
            sheet.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            widths = {"A": 14, "B": 50, "C": 55, "D": 9, "E": 9, "F": 9,
                      "G": 9, "H": 14, "I": 10, "J": 12, "K": 9,
                      "L": 10, "M": 8, "N": 30}
            for col, w in widths.items():
                ws.column_dimensions[col].width = w
            ws.freeze_panes = "C2"

        guide = pd.DataFrame(
            {
                "항목": [
                    "유형: 사실", "유형: 태도", "유형: 민감",
                    "포함", "배터리크기", "AB분할표본", "eta2_구간",
                ],
                "정의": [
                    "본인의 객관적 사실·행동을 묻는 문항 (투표 여부, 병원 방문 횟수, 가입 단체)",
                    "의견·평가·믿음을 묻는 문항 (정부 지출 의견, 기관 신뢰, 자긍심)",
                    "사회적으로 승인되는 답이 뚜렷한 문항 (이민자 인식, 성역할, 자살 태도, 지지 정당)",
                    "최종 30개에 넣을 후보면 Y. 각자 40개 내외로 표시할 것",
                    "같은 어간을 공유하는 문항 수. 같은 배터리에서 2개 이상 뽑으면 조건-타깃 누출 위험",
                    "같은 내용을 다른 문구로 물은 A/B형이 존재. 문구 효과의 인간 기준값이 있음",
                    "인구통계 설명력 3분위. 고=예측 가능성 높음, 저=인구통계만으로는 불가",
                ],
            }
        )
        guide.to_excel(xw, sheet_name="태깅기준", index=False)
        ws = xw.sheets["태깅기준"]
        ws.column_dimensions["A"].width = 18
        ws.column_dimensions["B"].width = 90

    print(f"-> {xlsx}")
    print("\n각자 시트에 독립적으로 태깅한 뒤 kgss_agreement.py로 일치도를 계산하세요.")

    # 요약 통계
    print("\n[배터리 상위 10]")
    print(size.head(10).to_string())
    if n_ab:
        print("\n[AB 분할표본 쌍]")
        for v, stem in sorted(ab_pairs.items()):
            lab = cand.loc[cand["var"] == v, "label"].iloc[0]
            print(f"  {v:16s} {lab[:50]}")


if __name__ == "__main__":
    main()