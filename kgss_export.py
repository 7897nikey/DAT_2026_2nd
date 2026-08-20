"""
KGSS .sav 파일을 사람이 직접 읽을 수 있는 형태로 변환

전체 23,282행 x 3,491열을 통째로 CSV로 빼면 8천만 셀이라 Excel이 사실상 열지 못한다.
따라서 두 가지로 나눠 내보낸다.

  1) 코드북 (codebook.xlsx)
     - 변수 사전: 변수명, 라벨, 척도, 선택지 목록, 회차별 조사 여부
     - "이 데이터에 무슨 문항이 있나"를 훑어보는 용도

  2) 회차별 데이터 (kgss_<year>_labeled.csv / _raw.csv)
     - 해당 회차에 실제 조사된 변수만, 숫자 코드 대신 라벨 텍스트로
     - 1,230행 x 500여 열이라 Excel에서 무리 없이 열린다

사용법:
    uv pip install openpyxl
    uv run kgss_export.py --sav "2003-2025_KGSS_kor_public_v2.sav" --year 2023
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

STRUCTURAL_CODES = {-8.0, -1.0}


def fmt_value_labels(vl: dict, limit: int = 12) -> str:
    """{1.0:'매우 그렇다', ...} -> '1=매우 그렇다 | 2=...' 형태로 정리."""
    if not vl:
        return ""
    items = sorted(vl.items(), key=lambda kv: float(kv[0]))
    parts = []
    for k, lab in items:
        code = float(k)
        if code in STRUCTURAL_CODES:
            continue
        code_s = str(int(code)) if code == int(code) else str(code)
        parts.append(f"{code_s}={lab}")
    if len(parts) > limit:
        parts = parts[:limit] + [f"...(총 {len(parts)}개)"]
    return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sav", required=True)
    ap.add_argument("--out", default="./export")
    ap.add_argument("--year", type=int, default=None, help="이 회차의 데이터를 내보냄")
    ap.add_argument("--encoding", default=None)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/4] 로드")
    df, meta = pyreadstat.read_sav(
        args.sav, apply_value_formats=False, encoding=args.encoding
    )

    print("[2/4] 결측 치환")
    clean = df.copy()
    for v in meta.column_names:
        clean[v] = clean[v].mask(clean[v].isin(list(STRUCTURAL_CODES)))

    print("[3/4] 코드북 생성")
    surveyed = clean.notna().groupby(df["YEAR"]).any().T  # 변수 x 연도 bool
    years = [int(c) for c in surveyed.columns]

    cb = pd.DataFrame(
        {
            "변수명": meta.column_names,
            "문항내용": [meta.column_names_to_labels.get(v, "") for v in meta.column_names],
            "척도": [meta.variable_measure.get(v, "") for v in meta.column_names],
            "선택지": [
                fmt_value_labels(meta.variable_value_labels.get(v, {}))
                for v in meta.column_names
            ],
        }
    )
    cb["조사회차수"] = surveyed.sum(axis=1).reindex(meta.column_names).values
    cb["조사연도"] = [
        ", ".join(str(y) for y, ok in zip(years, surveyed.loc[v]) if ok)
        if v in surveyed.index
        else ""
        for v in meta.column_names
    ]

    xlsx_path = outdir / "codebook.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
            cb.to_excel(xw, sheet_name="codebook", index=False)
            ws = xw.sheets["codebook"]
            for col, width in zip("ABCDEF", [16, 55, 10, 60, 12, 40]):
                ws.column_dimensions[col].width = width
            ws.freeze_panes = "A2"
        print(f"      -> {xlsx_path}")
    except ImportError:
        cb.to_csv(outdir / "codebook.csv", index=False, encoding="utf-8-sig")
        print("      openpyxl 없음 -> codebook.csv 로 저장")

    print("[4/4] 회차 데이터 내보내기")
    if args.year:
        m = df["YEAR"] == args.year
        sub = clean[m]
        keep = [v for v in sub.columns if sub[v].notna().any()]
        sub = sub[keep]
        print(f"      {args.year}년: {len(sub):,}행 x {len(keep)}열 (전체 {clean.shape[1]}열 중)")

        sub.to_csv(
            outdir / f"kgss_{args.year}_raw.csv", index=False, encoding="utf-8-sig"
        )

        lab = sub.copy()
        for v in keep:
            vl = meta.variable_value_labels.get(v, {})
            if not vl:
                continue
            mapping = {float(k): lab_ for k, lab_ in vl.items()}
            if lab[v].dtype.kind in "if":
                lab[v] = lab[v].map(mapping).fillna(lab[v])
        lab.to_csv(
            outdir / f"kgss_{args.year}_labeled.csv", index=False, encoding="utf-8-sig"
        )
        print(f"      -> kgss_{args.year}_raw.csv / kgss_{args.year}_labeled.csv")

    print(f"\n완료 -> {outdir.resolve()}")


if __name__ == "__main__":
    main()