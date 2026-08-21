r"""
값 라벨 vs 실제 출현 코드 대조

KOSSDA 통합 파일은 원본 회차의 결측 코드를 표준화한 것으로 보인다.
    원본 8 / 88 / 888 / 8888  (모름·무응답)  ->  통합 -8
    원본 9 / 99 / 999 / 9999  (비해당)       ->  통합 -1

그런데 .sav의 값 라벨에는 원본 코드와 통합 코드가 둘 다 남아 있다.
    SEX -> {-8: "DK", 1: "남자", 2: "여자", 8: "모르겠다/무응답"}

원본 코드가 실제로 데이터에 나타나는지, 아니면 잔여 라벨일 뿐인지
확인해야 결측 처리 규칙을 확정할 수 있다.

이 결과에 따라 kgss_inventory.py의 처리가 갈린다.
    8/88이 출현 안 함  -> -8을 DK로 재분류. STRUCTURAL은 -1만.
    8/88이 출현함      -> 회차별로 코드 체계가 다르므로 둘 다 처리해야 함

사용법:
    uv run kgss_code_audit.py --sav "2003-2025_KGSS_kor_public_v2.sav" --out .\inventory
"""

import argparse
import io
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DK_PAT = re.compile(
    r"(\bDK\b|Refus|모르겠다|모름|무응답|응답거부|거부|선택할\s*수\s*없)", re.I
)
INAP_PAT = re.compile(r"(Not Applicable|\bIAP\b|\binap\b|비해당|해당\s*없음)", re.I)
LEGACY_DK = {8.0, 88.0, 888.0, 8888.0}
LEGACY_INAP = {9.0, 99.0, 999.0, 9999.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sav", required=True)
    ap.add_argument("--out", default="./inventory")
    ap.add_argument("--encoding", default=None)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/3] 로드")
    df, meta = pyreadstat.read_sav(
        args.sav, apply_value_formats=False, encoding=args.encoding
    )
    print(f"      {len(df):,}행 x {len(meta.column_names):,}변수")

    print("[2/3] 라벨 등재 코드별 실제 출현 횟수")
    rows = []
    label_only, both = Counter(), Counter()
    for v in meta.column_names:
        vl = meta.variable_value_labels.get(v, {})
        if not vl:
            continue
        s = df[v]
        if s.dtype.kind not in "if":
            continue
        vc = s.value_counts()
        for k, lab in vl.items():
            code = float(k)
            n = int(vc.get(code, 0))
            kind = (
                "INAP" if INAP_PAT.search(str(lab))
                else "DK" if DK_PAT.search(str(lab))
                else "실질"
            )
            rows.append({
                "var": v, "code": code, "label": str(lab),
                "kind": kind, "n_occur": n,
                "legacy": code in LEGACY_DK | LEGACY_INAP,
            })
            if n == 0:
                label_only[code] += 1
            else:
                both[code] += 1
    audit = pd.DataFrame(rows)
    audit.to_csv(outdir / "code_audit.csv", index=False, encoding="utf-8-sig")

    print("[3/3] 요약")
    L = ["# 값 라벨 vs 실제 출현 코드\n"]

    # 결측 코드별 총 출현
    L.append("## 결측 관련 코드의 총 출현 횟수\n")
    L.append("| 코드 | 종류 | 라벨 등재 변수 | 실제 출현 변수 | 총 셀 수 |\n|---|---|---|---|---|")
    for code in sorted(LEGACY_DK | LEGACY_INAP | {-8.0, -1.0}):
        sub = audit[audit["code"] == code]
        if sub.empty:
            continue
        kinds = sub["kind"].mode()
        L.append(
            f"| {code:g} | {kinds.iloc[0] if len(kinds) else '-'} "
            f"| {len(sub):,} | {int((sub['n_occur'] > 0).sum()):,} "
            f"| {int(sub['n_occur'].sum()):,} |"
        )
    L.append("")

    legacy_used = audit[audit["legacy"] & (audit["n_occur"] > 0)]
    L.append(f"## 원본 코드(8/88/9/99 등)가 실제로 출현하는 변수: **{len(legacy_used):,}개**\n")
    if legacy_used.empty:
        L.append(
            "> 하나도 없습니다. 원본 코드는 잔여 라벨일 뿐이며, "
            "**실제 결측 코드는 -8(모름·무응답)과 -1(비해당) 두 개뿐**입니다.\n"
            "> -> kgss_inventory.py에서 -8을 STRUCTURAL이 아니라 DK로 재분류해야 합니다.\n"
        )
    else:
        L.append("| 변수 | 코드 | 라벨 | 출현 |\n|---|---|---|---|")
        for _, r in legacy_used.sort_values("n_occur", ascending=False).head(40).iterrows():
            L.append(f"| `{r['var']}` | {r['code']:g} | {r['label']} | {r['n_occur']:,} |")
        L.append("\n> 원본 코드가 살아 있습니다. 회차별로 코드 체계가 다를 수 있으므로 "
                 "연도별 출현 여부를 추가 확인해야 합니다.\n")

    # -8을 DK로 되살렸을 때 문항별 무응답률
    print("      -8 무응답률 상위 문항 계산")
    year_col = "YEAR"
    L.append("## -8(모름·무응답) 비율 상위 문항 — 2023년 조사분\n")
    if year_col in df.columns:
        y = df[df[year_col] == 2023]
        surveyed = [c for c in y.columns if (y[c] != -1).any() and y[c].dtype.kind in "if"]
        rates = []
        for c in surveyed:
            s = y[c]
            denom = int((s != -1).sum())
            if denom < 100:
                continue
            n8 = int((s == -8).sum())
            if n8:
                rates.append({
                    "var": c,
                    "label": meta.column_names_to_labels.get(c, ""),
                    "n_dk": n8, "pct_dk": round(n8 / denom * 100, 2),
                })
        rt = pd.DataFrame(rates).sort_values("pct_dk", ascending=False)
        rt.to_csv(outdir / "dk_rate_2023.csv", index=False, encoding="utf-8-sig")
        L.append(f"- 무응답이 하나라도 있는 문항: **{len(rt):,}개**")
        if not rt.empty:
            L.append(f"- 무응답률 중앙값 **{rt['pct_dk'].median():.2f}%**\n")
            L.append("| 문항 | 라벨 | 무응답 | 비율 |\n|---|---|---|---|")
            for _, r in rt.head(25).iterrows():
                L.append(f"| `{r['var']}` | {r['label'][:45]} | {r['n_dk']} | {r['pct_dk']}% |")
        L.append(
            "\n> 무응답률이 높은 문항일수록 '실제 분포에 무응답을 포함할지'가 "
            "결과를 크게 바꿉니다. LLM은 '모르겠다'를 거의 내지 않기 때문입니다.\n"
        )

    (outdir / "code_audit.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n완료 -> {outdir.resolve()}")
    print("  code_audit.csv / dk_rate_2023.csv / code_audit.md")


if __name__ == "__main__":
    main()