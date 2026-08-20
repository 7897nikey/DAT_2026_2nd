r"""
KGSS 2003-2025 누적자료 구조 진단

결측 4층:
    STRUCTURAL : -1, -8           해당 회차 미조사 -> 항상 제외
    INAP       : IAP / 9/99/999   비해당(스킵) -> 항상 제외
    DK         : 8/88/888         모름·무응답 -> 결과 문항은 유지(설계 선택),
                                  인구통계 변수는 항상 제외
    실질 응답

핵심 설계 판단:
  1) DK/INAP 판정은 라벨 어휘만으로 하면 안 된다.
     ELEFRAUDA의 2번 선택지 "부정선거가 있었는지 없었는지 모르겠다"는
     응답자가 적극적으로 고른 실질 응답이지 무응답이 아니다.
     코드값이 실질 선택지의 연속 구간(0 또는 1에서 시작) 안에 있으면
     라벨과 무관하게 실질 응답으로 취급한다.
  2) 인구통계 셀 변수는 pd.cut이 아니라 명시적 매핑으로 파생한다.
     EDUC은 0=무학, 8=서당한학, 88=DK가 섞여 있어 구간 절단이 오류를 낸다.
  3) 순서형 여부는 SPSS measure 속성이 아니라 값 라벨 어휘로 추론한다.
     measure는 작성자 임의 지정 값이라 리커트 배터리가 nominal로 찍힌다.
  4) A/B 분할표본 문항은 응답률이 약 50%가 정상이므로
     일반 문항의 커버리지 기준으로 거르면 안 된다.

사용법:
    uv run kgss_inventory.py --sav "2003-2025_KGSS_kor_public_v2.sav" ^
        --out .\inventory --target-year 2023
"""

import argparse
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------- 결측 코드
STRUCTURAL_CODES = {-8.0, -1.0}
NEG1_LEGIT: set[str] = set()

DK_PAT = re.compile(
    r"(\bDK\b|Refus|모르겠다|모름|무응답|응답거부|거부|don.?t know|no answer)", re.I
)
INAP_PAT = re.compile(r"(Not Applicable|\bIAP\b|\binap\b|비해당|해당\s*없음)", re.I)

# ---------------------------------------------------------------- 인구통계
DEMO_VARS = ["SEX", "AGE", "EDUC", "MARITAL", "REGION", "URBAN", "INCOME", "RINCOME"]

AGE_BINS = [0, 29, 39, 49, 59, 69, 200]
# EDUC: 0=무학 1=초등 2=중학 3=고교 4=전문대 5=대학 6=석사 7=박사 8=서당한학 88=DK
EDU_MAP = {0: 0, 1: 0, 2: 0, 3: 0, 8: 0, 4: 1, 5: 1, 6: 1, 7: 1}
EDU_LABELS = ["고졸 이하", "대졸 이상"]

# ---------------------------------------------------------------- 순서형 추론
ORDINAL_LEXICON = [
    r"매우", r"약간", r"대체로", r"별로", r"전혀", r"조금", r"아주", r"다소",
    r"그렇다", r"아니다", r"찬성", r"반대", r"동의",
    r"만족", r"불만족", r"많이", r"적게", r"높", r"낮",
    r"항상", r"자주", r"가끔", r"거의", r"드물", r"없다", r"없음",
    r"늘려야", r"줄여야", r"현재대로", r"좋아", r"나빠",
    r"신뢰", r"중요", r"심각", r"쉽", r"어렵",
]
ORDINAL_PAT = re.compile("|".join(ORDINAL_LEXICON))
AB_FORM_PAT = re.compile(r"\(\s*[AB]\s*형\s*\)")
PROXY_PREFIX = re.compile(r"^(SP|PA|MA|FA|MO)[A-Z]")


def substantive_run(codes) -> set[float]:
    """실질 선택지의 연속 구간을 찾는다.

    0 또는 1에서 시작해 1씩 증가하는 구간만 실질로 인정한다.
    AGE처럼 값 라벨이 888/999뿐이면 구간이 없으므로 전부 결측 처리된다.
    """
    pos = sorted(c for c in codes if c not in STRUCTURAL_CODES and c >= 0)
    if not pos or pos[0] not in (0.0, 1.0):
        return set()
    run = [pos[0]]
    for c in pos[1:]:
        if c == run[-1] + 1:
            run.append(c)
        else:
            break
    return set(run)


def classify_missing(value_labels: dict) -> dict:
    """결측 코드를 DK / INAP으로 분류.

    라벨 어휘만으로 판정하지 않는다. 실질 선택지 연속 구간 안의 코드는
    "모르겠다" 같은 어휘가 붙어 있어도 실질 응답으로 본다.
    """
    codes = {float(k) for k in value_labels}
    run = substantive_run(codes)
    out = {"dk": [], "inap": []}
    for k, lab in value_labels.items():
        code, lab = float(k), str(lab)
        if code in STRUCTURAL_CODES or code in run:
            continue
        if INAP_PAT.search(lab):
            out["inap"].append(code)
        elif DK_PAT.search(lab):
            out["dk"].append(code)
    return out


def guess_ordinal(value_labels: dict, dk: list, inap: list) -> bool:
    subs = [
        str(lab)
        for k, lab in value_labels.items()
        if float(k) not in STRUCTURAL_CODES
        and float(k) not in dk
        and float(k) not in inap
    ]
    if not (3 <= len(subs) <= 7):
        return False
    hits = sum(bool(ORDINAL_PAT.search(s)) for s in subs)
    return hits >= max(2, len(subs) // 2)


def eta_squared(d: pd.DataFrame, item: str, cells: list[str]) -> dict | None:
    d = d[[item] + cells].dropna()
    if len(d) < 100:
        return None
    g = d.groupby(cells, observed=True)[item]
    gm = d[item].mean()
    ss_total = ((d[item] - gm) ** 2).sum()
    if ss_total <= 0:
        return None
    ss_between = (g.count() * (g.mean() - gm) ** 2).sum()
    e2 = float(ss_between / ss_total)
    return {
        "item": item, "n": len(d), "n_cells": int(g.ngroups),
        "min_cell_n": int(g.count().min()),
        "eta2": round(e2, 4), "unexplained": round(1 - e2, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sav", required=True)
    ap.add_argument("--out", default="./inventory")
    ap.add_argument("--encoding", default=None)
    ap.add_argument("--target-year", type=int, default=None)
    ap.add_argument("--drop-dk", action="store_true")
    ap.add_argument("--cells", nargs="+", default=None, help="지정 없으면 자동 선택")
    ap.add_argument("--min-cov", type=float, default=0.9)
    ap.add_argument("--ab-min-cov", type=float, default=0.4, help="A/B 문항 기준")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/8] 로드")
    df, meta = pyreadstat.read_sav(
        args.sav, apply_value_formats=False, encoding=args.encoding
    )
    print(f"      {len(df):,}행 x {len(meta.column_names):,}변수")

    print("[2/8] sentinel 출현")
    numeric = df.select_dtypes(include=[np.number])
    total_cells = numeric.size
    sent = {c: int((numeric == c).sum().sum()) for c in sorted(STRUCTURAL_CODES)}
    for c, n in sent.items():
        print(f"      {c}: {n:,}셀 ({n/total_cells*100:.1f}%)")

    print("[3/8] 변수 사전 + 결측 분류 + 순서형 추론")
    rows, dk_map, inap_map = [], {}, {}
    rescued = []
    for v in meta.column_names:
        vl = meta.variable_value_labels.get(v, {})
        c = classify_missing(vl)
        dk_map[v], inap_map[v] = c["dk"], c["inap"]
        # 구제된 선택지(라벨은 DK/INAP 같지만 실질 구간 안) 기록
        run = substantive_run({float(k) for k in vl})
        for k, lab in vl.items():
            if float(k) in run and (DK_PAT.search(str(lab)) or INAP_PAT.search(str(lab))):
                rescued.append({"var": v, "code": float(k), "label": str(lab)})
        rows.append({
            "var": v,
            "label": meta.column_names_to_labels.get(v, ""),
            "measure": meta.variable_measure.get(v, ""),
            "ordinal_guess": guess_ordinal(vl, c["dk"], c["inap"]),
            "dk": json.dumps(c["dk"]),
            "inap": json.dumps(c["inap"]),
            "value_labels": json.dumps(vl, ensure_ascii=False),
            "is_proxy": bool(PROXY_PREFIX.match(v)),
        })
    vt = pd.DataFrame(rows)
    vt.to_csv(outdir / "variables.csv", index=False, encoding="utf-8-sig")
    resc = pd.DataFrame(rescued)
    resc.to_csv(outdir / "rescued_options.csv", index=False, encoding="utf-8-sig")
    n_guess = int(vt["ordinal_guess"].sum())
    print(f"      순서형 추론 {n_guess:,}개 / 실질 응답으로 구제된 선택지 {len(resc):,}개")

    print("[4/8] 결측 치환")
    clean = df.copy()
    for v in meta.column_names:
        codes = set(STRUCTURAL_CODES)
        if v in NEG1_LEGIT:
            codes.discard(-1.0)
        drop = list(codes) + list(inap_map[v])
        if args.drop_dk or v in DEMO_VARS:
            drop += list(dk_map[v])
        clean[v] = clean[v].mask(clean[v].isin(drop))
    clean.to_parquet(outdir / "kgss_clean.parquet", index=False)

    print("[5/8] 커버리지 행렬")
    cov = clean.notna().groupby(df["YEAR"]).mean().T
    cov.columns = [f"y{int(c)}" for c in cov.columns]
    ycols = list(cov.columns)
    cov.index.name = "var"
    cov["n_years"] = (cov[ycols] > 0.5).sum(axis=1)
    cov.insert(0, "label", [meta.column_names_to_labels.get(v, "") for v in cov.index])
    cov.reset_index().to_csv(
        outdir / "year_coverage.csv", index=False, encoding="utf-8-sig"
    )
    n_waves = len(ycols)
    common = int((cov["n_years"] >= n_waves - 1).sum())
    single = int((cov["n_years"] == 1).sum())
    print(f"      전 회차 공통 {common} / 단일 회차 {single}")

    print("[6/8] FINALWT")
    wt = df.groupby("YEAR")["FINALWT"].agg(["count", "sum", "min", "max"])
    wt["sum_over_n"] = (wt["sum"] / wt["count"]).round(4)

    print("[7/8] 문항 후보 + 셀 진단 + eta2")
    n_cand = n_surveyed = n_ab = None
    cellcheck = eta = chosen = None
    if args.target_year:
        y = f"y{args.target_year}"
        sub = clean[df["YEAR"] == args.target_year].copy()
        n_surveyed = int((sub.notna().sum() > 0).sum())

        sub["AGE_G"] = pd.cut(sub["AGE"], AGE_BINS, labels=False)
        sub["EDU_G"] = sub["EDUC"].map(EDU_MAP)
        for c in ("AGE_G", "EDU_G"):
            print(f"      {c} 결측 {int(sub[c].isna().sum())}명 / {len(sub)}명")

        configs = [
            ["SEX", "AGE_G"],
            ["SEX", "AGE_G", "EDU_G"],
            ["SEX", "AGE_G", "EDU_G", "URBAN"],
            ["SEX", "AGE_G", "EDU_G", "REGION"],
        ]
        cc = []
        for cfg in configs:
            if not all(c in sub for c in cfg):
                continue
            g = sub.dropna(subset=cfg).groupby(cfg, observed=True).size()
            cc.append({
                "cells": " x ".join(cfg), "n_cells": len(g),
                "min_n": int(g.min()), "median_n": float(g.median()),
                "pct_ge20": round(float((g >= 20).mean()) * 100, 1),
                "config": json.dumps(cfg),
            })
        cellcheck = pd.DataFrame(cc)
        cellcheck.to_csv(outdir / "cell_size_check.csv", index=False, encoding="utf-8-sig")

        if args.cells:
            chosen = args.cells
        else:
            ok = cellcheck[cellcheck["pct_ge20"] >= 80]
            chosen = json.loads(ok.iloc[-1]["config"]) if not ok.empty else ["SEX", "AGE_G"]
        print(f"      선택된 셀 구성: {' x '.join(chosen)}")

        # A/B 분할표본 문항은 커버리지 기준을 따로 적용
        info = vt.set_index("var")
        ab_vars = set(
            info.index[info["label"].fillna("").str.contains(AB_FORM_PAT, regex=True)]
        )
        is_ab = cov.index.isin(ab_vars)
        wave = cov[(cov[y] > args.min_cov) | (is_ab & (cov[y] > args.ab_min_cov))].copy()
        wave = wave[~wave.index.to_series().str.match(PROXY_PREFIX)]
        wave["is_ab"] = wave.index.isin(ab_vars)
        wave["n_options"] = sub[wave.index].nunique()
        wave["measure"] = info["measure"].reindex(wave.index)
        wave["ordinal_guess"] = info["ordinal_guess"].reindex(wave.index)
        wave["is_item_cand"] = wave["n_options"].between(2, 7) & (
            ~wave.index.isin(DEMO_VARS)
        )
        wave.reset_index()[
            ["var", "label", y, "n_years", "n_options", "measure",
             "ordinal_guess", "is_ab", "is_item_cand"]
        ].to_csv(outdir / "wave_items.csv", index=False, encoding="utf-8-sig")
        n_cand = int(wave["is_item_cand"].sum())
        n_ab = int((wave["is_ab"] & wave["is_item_cand"]).sum())
        print(f"      문항 후보 {n_cand}개 (A/B 분할표본 {n_ab}개 포함)")

        items = wave.index[wave["is_item_cand"] & wave["ordinal_guess"]].tolist()
        recs = [r for it in items if (r := eta_squared(sub, it, chosen))]
        eta = pd.DataFrame(recs)
        if not eta.empty:
            eta["label"] = [meta.column_names_to_labels.get(i, "") for i in eta["item"]]
            eta = eta.sort_values("eta2", ascending=False)
            eta.to_csv(outdir / "eta2.csv", index=False, encoding="utf-8-sig")

    print("[8/8] 요약 작성")
    L = ["# KGSS 구조 진단\n"]
    L.append(f"- 관측치 {len(df):,}행 / 변수 {len(meta.column_names):,}개 / 회차 {n_waves}개\n")

    L.append("## sentinel\n| 코드 | 셀 수 | 비율 |\n|---|---|---|")
    for c, n in sent.items():
        L.append(f"| {c} | {n:,} | {n/total_cells*100:.1f}% |")
    L.append("")

    L.append("## 문항 분포\n")
    L.append(f"- 거의 전 회차 공통: **{common}개** / 단일 회차 전용: **{single}개**")
    L.append(f"- 순서형 추론: **{n_guess:,}개**")
    L.append(f"- 실질 응답으로 구제된 선택지: **{len(resc):,}개** (rescued_options.csv)\n")
    if not resc.empty:
        L.append("> 라벨에 '모르겠다' 등이 있으나 실질 선택지 구간 안이라 "
                 "응답으로 유지한 항목입니다. 예: ELEFRAUDA의 "
                 "'부정선거가 있었는지 없었는지 모르겠다'\n")
        L.append("| 변수 | 코드 | 라벨 |\n|---|---|---|")
        for _, r in resc.head(20).iterrows():
            L.append(f"| `{r['var']}` | {r['code']:g} | {r['label']} |")
        L.append("")

    L.append("## FINALWT\n| 연도 | N | 합/N | 최소 | 최대 |\n|---|---|---|---|---|")
    for y_, r in wt.iterrows():
        L.append(f"| {int(y_)} | {int(r['count']):,} | {r['sum_over_n']} "
                 f"| {r['min']:.3f} | {r['max']:.3f} |")
    L.append("")

    if n_cand is not None:
        L.append(f"## {args.target_year}년 회차\n")
        L.append(f"- 실제 조사된 변수: **{n_surveyed}개** / {len(meta.column_names):,}개")
        L.append(f"- 문항 후보: **{n_cand}개** (A/B 분할표본 {n_ab}개 포함)")
        L.append(f"- 선택된 셀 구성: **{' x '.join(chosen)}**\n")

        L.append("### 셀 구성별 표본 크기\n")
        L.append("| 셀 구성 | 셀 수 | 최소 n | 중앙 n | n>=20 |\n|---|---|---|---|---|")
        for _, r in cellcheck.iterrows():
            L.append(f"| {r['cells']} | {r['n_cells']} | {r['min_n']} "
                     f"| {r['median_n']:.0f} | {r['pct_ge20']}% |")
        L.append("")

        if eta is not None and not eta.empty:
            L.append("### eta2\n")
            L.append(f"- 대상 **{len(eta)}문항** / 중앙값 **{eta['eta2'].median():.3f}**")
            L.append(f"- 인구통계가 설명하는 응답 분산 중앙값 "
                     f"**{eta['eta2'].median()*100:.1f}%**\n")
            for title, part in [("설명력 상위 15", eta.head(15)),
                                ("설명력 하위 15", eta.tail(15).iloc[::-1])]:
                L.append(f"#### {title}\n| 문항 | 라벨 | eta2 |\n|---|---|---|")
                for _, r in part.iterrows():
                    L.append(f"| `{r['item']}` | {r['label']} | {r['eta2']} |")
                L.append("")

    (outdir / "diagnostics.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n완료 -> {outdir.resolve()}")


if __name__ == "__main__":
    main()