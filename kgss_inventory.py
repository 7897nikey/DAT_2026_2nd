r"""
KGSS 2003-2025 누적자료 구조 진단

결측 코드
    -1  비해당      회차 미조사 + 스킵 패턴 비해당이 섞여 있다.
                    그 해에 -1이 아닌 값이 하나라도 있으면 조사된 문항이고,
                    그 안의 -1은 스킵 비해당이다.
    -8  모름·무응답  실제 조사에서 '모르겠다'는 선택지로 제시되지 않았고
                    자발적 응답만 기록됐다. 따라서 제거가 기본값이며,
                    --keep-dk로 민감도 분석이 가능하다.
    값 라벨의 8 / 88 / 9 / 99는 원본 회차의 잔여물이다. 실제 출현하는 8, 9는
    대부분 실질 응답이므로(ATTEND 8='전혀 가지 않는다'), 코드값이 아니라
    실질 선택지의 연속 구간으로 판정한다.

이 스크립트가 산출하는 것과 그 용도

    modal_baseline.csv   셀 최빈응답 정확도.
                         인구통계만 주어졌을 때 도달 가능한 개인 예측의 상한이다.
                         eta2는 분산 기반이라 명목형에 정의되지 않고 정확도와
                         단위가 다르므로, 상한 논의에는 이쪽을 쓴다.
    eta2.csv             인구통계 설명력. 문항 선정의 층화 축으로만 쓰고,
                         재현도와의 관계는 검증할 가설로 둔다.
    conditional_items.csv 스킵 구조가 있어 주 문항에서 제외된 문항.
                         비해당 응답자를 대상으로 '스킵 준수율'을 잴 수 있다.
    common_items.csv     거의 전 회차에 등장하는 문항. 회차 간 검증 가능성 판단용.

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

IAP_CODE = -1.0
DK_CODE = -8.0
NEG1_LEGIT: set[str] = set()

DK_PAT = re.compile(
    r"(\bDK\b|Refus|모르겠다|모름|무응답|응답거부|거부|선택할\s*수\s*없|"
    r"don.?t know|no answer)", re.I)
INAP_PAT = re.compile(r"(Not Applicable|\bIAP\b|\binap\b|비해당|해당\s*없음)", re.I)

DEMO_VARS = ["SEX", "AGE", "EDUC", "MARITAL", "REGION", "URBAN", "INCOME", "RINCOME"]
AGE_BINS = [0, 29, 39, 49, 59, 69, 200]
EDU_MAP = {0: 0, 1: 0, 2: 0, 3: 0, 8: 0, 4: 1, 5: 1, 6: 1, 7: 1}

ORDINAL_LEXICON = [
    r"매우", r"약간", r"대체로", r"별로", r"전혀", r"조금", r"아주", r"다소",
    r"그렇다", r"아니다", r"찬성", r"반대", r"동의", r"만족", r"불만족",
    r"많이", r"적게", r"높", r"낮", r"항상", r"자주", r"가끔", r"거의", r"드물",
    r"없다", r"없음", r"늘려야", r"줄여야", r"현재대로", r"좋아", r"나빠",
    r"신뢰", r"중요", r"심각", r"쉽", r"어렵",
]
ORDINAL_PAT = re.compile("|".join(ORDINAL_LEXICON))
AB_FORM_PAT = re.compile(r"\(\s*[AB]\s*형\s*\)")
PROXY_PREFIX = re.compile(r"^(SP|PA|MA|FA|MO)[A-Z]")
BATTERY_TAIL = re.compile(r"\d+$")


def substantive_run(codes) -> set[float]:
    pos = sorted(c for c in codes if c >= 0)
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
    codes = {float(k) for k in value_labels}
    run = substantive_run(codes)
    out = {"dk": [], "inap": []}
    for k, lab in value_labels.items():
        code, lab = float(k), str(lab)
        if code in (IAP_CODE, DK_CODE) or code in run:
            continue
        if INAP_PAT.search(lab):
            out["inap"].append(code)
        elif DK_PAT.search(lab):
            out["dk"].append(code)
    return out


def guess_ordinal(value_labels: dict, dk: list, inap: list) -> bool:
    drop = {IAP_CODE, DK_CODE} | set(dk) | set(inap)
    subs = [str(v) for k, v in value_labels.items() if float(k) not in drop]
    if not (3 <= len(subs) <= 7):
        return False
    return sum(bool(ORDINAL_PAT.search(s)) for s in subs) >= max(2, len(subs) // 2)


def eta_squared(d: pd.DataFrame, item: str, cells: list[str]) -> dict | None:
    d = d[[item] + cells].dropna()
    if len(d) < 100:
        return None
    g = d.groupby(cells, observed=True)[item]
    gm = d[item].mean()
    sst = ((d[item] - gm) ** 2).sum()
    if sst <= 0:
        return None
    ssb = (g.count() * (g.mean() - gm) ** 2).sum()
    e2 = float(ssb / sst)
    return {"item": item, "eta2": round(e2, 4), "unexplained": round(1 - e2, 4)}


def modal_baseline(d: pd.DataFrame, item: str, cells: list[str]) -> dict | None:
    """인구통계만으로 도달 가능한 개인 예측 정확도의 상한.

    marginal_modal : 전체 최빈 응답을 항상 찍었을 때의 정확도 (인구통계 무시)
    cell_modal     : 각 셀의 최빈 응답을 찍었을 때의 정확도 (인구통계 최대 활용)
    gain           : 인구통계를 알아서 얻는 순이익

    LLM 정확도는 marginal_modal을 넘어야 의미가 있고,
    cell_modal을 넘으면 인구통계 이상의 무언가를 쓰고 있다는 뜻이다.
    eta2와 달리 명목형 문항에도 정의된다.
    """
    d = d[[item] + cells].dropna()
    if len(d) < 100:
        return None
    marginal = float(d[item].value_counts(normalize=True).max())
    acc = 0.0
    for _, s in d.groupby(cells, observed=True)[item]:
        acc += len(s) / len(d) * float(s.value_counts(normalize=True).max())
    return {"item": item, "n": len(d),
            "marginal_modal": round(marginal, 4),
            "cell_modal": round(acc, 4),
            "gain": round(acc - marginal, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sav", required=True)
    ap.add_argument("--out", default="./inventory")
    ap.add_argument("--encoding", default=None)
    ap.add_argument("--target-year", type=int, default=None)
    ap.add_argument("--keep-dk", action="store_true")
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--min-answer", type=float, default=0.9)
    ap.add_argument("--max-iap", type=float, default=0.05)
    ap.add_argument("--ab-min-answer", type=float, default=0.4)
    ap.add_argument("--common-slack", type=int, default=1,
                    help="전 회차 공통 판정 시 허용할 결번 회차 수")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/9] 로드")
    df, meta = pyreadstat.read_sav(
        args.sav, apply_value_formats=False, encoding=args.encoding)
    print(f"      {len(df):,}행 x {len(meta.column_names):,}변수")

    print("[2/9] 결측 코드 출현")
    num = df.select_dtypes(include=[np.number])
    total = num.size
    n_iap = int((num == IAP_CODE).sum().sum())
    n_dk = int((num == DK_CODE).sum().sum())
    print(f"      -1 {n_iap:,}셀 ({n_iap/total*100:.1f}%) / "
          f"-8 {n_dk:,}셀 ({n_dk/total*100:.2f}%)")

    print("[3/9] 변수 사전")
    rows, dk_map, inap_map = [], {}, {}
    for v in meta.column_names:
        vl = meta.variable_value_labels.get(v, {})
        c = classify_missing(vl)
        dk_map[v], inap_map[v] = c["dk"], c["inap"]
        rows.append({
            "var": v,
            "label": meta.column_names_to_labels.get(v, ""),
            "measure": meta.variable_measure.get(v, ""),
            "ordinal_guess": guess_ordinal(vl, c["dk"], c["inap"]),
            "battery": BATTERY_TAIL.sub("", v) or v,
            "legacy_dk": json.dumps(c["dk"]),
            "legacy_inap": json.dumps(c["inap"]),
            "value_labels": json.dumps(vl, ensure_ascii=False),
            "is_ab": bool(AB_FORM_PAT.search(meta.column_names_to_labels.get(v, "") or "")),
            "is_proxy": bool(PROXY_PREFIX.match(v)),
        })
    vt = pd.DataFrame(rows)
    vt.to_csv(outdir / "variables.csv", index=False, encoding="utf-8-sig")
    info = vt.set_index("var")
    print(f"      순서형 추론 {int(vt['ordinal_guess'].sum()):,}개")

    print("[4/9] 회차별 결측 분해")
    years = sorted(df["YEAR"].unique())
    cols = [v for v in meta.column_names if df[v].dtype.kind in "if"]
    yg = df["YEAR"]
    recs = []
    for v in cols:
        s = df[v]
        is_iap = (s == IAP_CODE)
        if inap_map[v]:
            is_iap = is_iap | s.isin(inap_map[v])
        is_dk = (s == DK_CODE)
        if dk_map[v]:
            is_dk = is_dk | s.isin(dk_map[v])
        is_ans = ~(is_iap | is_dk | s.isna())
        g = pd.DataFrame({"iap": is_iap, "dk": is_dk, "ans": is_ans}).groupby(yg).mean()
        for y in years:
            recs.append({"var": v, "year": int(y),
                         "iap_rate": round(float(g.at[y, "iap"]), 4),
                         "dk_rate": round(float(g.at[y, "dk"]), 4),
                         "answer_rate": round(float(g.at[y, "ans"]), 4)})
    lr = pd.DataFrame(recs)
    lr["in_wave"] = lr["answer_rate"] + lr["dk_rate"] > 0
    lr.to_csv(outdir / "missing_by_year.csv", index=False, encoding="utf-8-sig")

    n_waves = len(years)
    wave_cnt = lr.groupby("var")["in_wave"].sum()

    print("[5/9] 전 회차 공통 문항")
    thr = n_waves - args.common_slack
    common_vars = wave_cnt[wave_cnt >= thr].index.tolist()
    cm = info.reindex(common_vars)[["label", "measure", "ordinal_guess", "battery", "is_proxy"]]
    cm["n_waves"] = wave_cnt.reindex(common_vars)
    cm["n_options"] = [int(df[v][~df[v].isin([IAP_CODE, DK_CODE])].nunique()) for v in common_vars]
    cm["is_demo"] = cm.index.isin(DEMO_VARS)
    cm["is_item_cand"] = (cm["n_options"].between(2, 7) & ~cm["is_demo"] & ~cm["is_proxy"])
    cm.reset_index().to_csv(outdir / "common_items.csv", index=False, encoding="utf-8-sig")
    n_common_cand = int(cm["is_item_cand"].sum())
    n_common_ord = int((cm["is_item_cand"] & cm["ordinal_guess"]).sum())
    print(f"      {thr}개 회차 이상 등장: {len(common_vars)}개 "
          f"(문항 후보 {n_common_cand}, 그중 순서형 {n_common_ord})")

    print("[6/9] 결측 치환")
    clean = df.copy()
    for v in meta.column_names:
        drop = [] if v in NEG1_LEGIT else [IAP_CODE]
        drop += list(inap_map[v])
        if (not args.keep_dk) or v in DEMO_VARS:
            drop += [DK_CODE] + list(dk_map[v])
        clean[v] = clean[v].mask(clean[v].isin(drop))
    clean.to_parquet(outdir / "kgss_clean.parquet", index=False)
    print(f"      DK 처리: {'유지' if args.keep_dk else '제거'}")

    print("[7/9] FINALWT")
    wt = df.groupby("YEAR")["FINALWT"].agg(["count", "sum", "min", "max"])
    wt["sum_over_n"] = (wt["sum"] / wt["count"]).round(4)

    print("[8/9] 문항 후보 / 베이스라인")
    n_cand = n_ab = n_surveyed = n_cond = None
    cellcheck = base = eta = chosen = None
    ty = args.target_year
    if ty:
        sub = clean[df["YEAR"] == ty].copy()
        ry = lr[lr["year"] == ty].set_index("var")
        n_surveyed = int(ry["in_wave"].sum())

        sub["AGE_G"] = pd.cut(sub["AGE"], AGE_BINS, labels=False)
        sub["EDU_G"] = sub["EDUC"].map(EDU_MAP)

        cc = []
        for cfg in [["SEX", "AGE_G"], ["SEX", "AGE_G", "EDU_G"],
                    ["SEX", "AGE_G", "EDU_G", "URBAN"],
                    ["SEX", "AGE_G", "EDU_G", "REGION"]]:
            if not all(c in sub for c in cfg):
                continue
            g = sub.dropna(subset=cfg).groupby(cfg, observed=True).size()
            cc.append({"cells": " x ".join(cfg), "n_cells": len(g),
                       "min_n": int(g.min()), "median_n": float(g.median()),
                       "pct_ge20": round(float((g >= 20).mean()) * 100, 1),
                       "config": json.dumps(cfg)})
        cellcheck = pd.DataFrame(cc)
        cellcheck.to_csv(outdir / "cell_size_check.csv", index=False, encoding="utf-8-sig")
        if args.cells:
            chosen = args.cells
        else:
            ok = cellcheck[cellcheck["pct_ge20"] >= 80]
            chosen = json.loads(ok.iloc[-1]["config"]) if not ok.empty else ["SEX", "AGE_G"]
        print(f"      셀 구성: {' x '.join(chosen)}")

        w = ry.copy()
        w["label"] = info["label"].reindex(w.index)
        w["is_ab"] = info["is_ab"].reindex(w.index).fillna(False)
        w["ordinal_guess"] = info["ordinal_guess"].reindex(w.index)
        w["battery"] = info["battery"].reindex(w.index)
        w = w[w["in_wave"] & ~w.index.to_series().str.match(PROXY_PREFIX)]
        w["n_options"] = sub[w.index].nunique()

        ans_thr = np.where(w["is_ab"], args.ab_min_answer, args.min_answer)
        w["pass_answer"] = w["answer_rate"] > ans_thr
        w["pass_iap"] = w["iap_rate"] <= args.max_iap
        w["is_item_cand"] = (w["pass_answer"] & w["pass_iap"]
                             & w["n_options"].between(2, 7)
                             & ~w.index.isin(DEMO_VARS))
        w.reset_index().to_csv(outdir / "wave_items.csv", index=False, encoding="utf-8-sig")
        n_cand = int(w["is_item_cand"].sum())
        n_ab = int((w["is_ab"] & w["is_item_cand"]).sum())

        # 조건부 문항: 조사됐고 선택지도 적절하나 스킵 비해당이 많은 것
        cond = w[(~w["pass_iap"]) & w["n_options"].between(2, 7)
                 & ~w.index.isin(DEMO_VARS)].copy()
        cond["n_eligible"] = (sub[cond.index].notna().sum()
                              if len(cond) else pd.Series(dtype=int))
        cond["n_inapplicable"] = (len(sub) - cond["n_eligible"]).astype(int)
        cond.sort_values("iap_rate", ascending=False).reset_index().to_csv(
            outdir / "conditional_items.csv", index=False, encoding="utf-8-sig")
        n_cond = len(cond)
        print(f"      문항 후보 {n_cand}개 (A/B {n_ab}) / 조건부 문항 {n_cond}개")

        items = w.index[w["is_item_cand"]].tolist()
        base = pd.DataFrame([r for it in items if (r := modal_baseline(sub, it, chosen))])
        if not base.empty:
            base["label"] = [meta.column_names_to_labels.get(i, "") for i in base["item"]]
            base = base.sort_values("gain", ascending=False)
            base.to_csv(outdir / "modal_baseline.csv", index=False, encoding="utf-8-sig")

        ord_items = w.index[w["is_item_cand"] & w["ordinal_guess"]].tolist()
        eta = pd.DataFrame([r for it in ord_items if (r := eta_squared(sub, it, chosen))])
        if not eta.empty:
            eta["label"] = [meta.column_names_to_labels.get(i, "") for i in eta["item"]]
            eta["dk_rate"] = ry["dk_rate"].reindex(eta["item"]).values
            eta = eta.sort_values("eta2", ascending=False)
            eta.to_csv(outdir / "eta2.csv", index=False, encoding="utf-8-sig")

    print("[9/9] 요약")
    L = ["# KGSS 구조 진단\n"]
    L.append(f"- 관측치 {len(df):,}행 / 변수 {len(meta.column_names):,}개 / 회차 {n_waves}개")
    L.append(f"- DK 처리: **{'유지' if args.keep_dk else '제거'}** (인구통계는 항상 제거)\n")

    L.append("## 결측 코드\n| 코드 | 뜻 | 셀 수 | 비율 |\n|---|---|---|---|")
    L.append(f"| -1 | 비해당(미조사 + 스킵) | {n_iap:,} | {n_iap/total*100:.1f}% |")
    L.append(f"| -8 | 모름·무응답 | {n_dk:,} | {n_dk/total*100:.2f}% |")
    L.append("")

    L.append(f"## 전 회차 공통 문항 ({thr}개 회차 이상)\n")
    L.append(f"- 전체 **{len(common_vars)}개**")
    L.append(f"- 문항 후보(선택지 2~7개, 인구통계·대리응답 제외): **{n_common_cand}개**")
    L.append(f"- 그중 순서형: **{n_common_ord}개**\n")
    L.append("> 회차 간 검증(2회차 → 17회차 확장)이 가능한지 판단하는 근거입니다. "
             "태도·민감 문항이 충분해야 성립합니다. 목록은 common_items.csv 참조.\n")
    if n_common_cand:
        cand_cm = cm[cm["is_item_cand"]].head(30)
        L.append("| 문항 | 라벨 | 회차 | 선택지 | 순서형 |\n|---|---|---|---|---|")
        for v, r in cand_cm.iterrows():
            L.append(f"| `{v}` | {str(r['label'])[:40]} | {int(r['n_waves'])} "
                     f"| {int(r['n_options'])} | {'O' if r['ordinal_guess'] else '-'} |")
        L.append("")

    L.append("## FINALWT\n| 연도 | N | 합/N | 최소 | 최대 |\n|---|---|---|---|---|")
    for y_, r in wt.iterrows():
        L.append(f"| {int(y_)} | {int(r['count']):,} | {r['sum_over_n']} "
                 f"| {r['min']:.3f} | {r['max']:.3f} |")
    L.append("")

    if ty:
        L.append(f"## {ty}년 회차\n")
        L.append(f"- 조사된 문항 **{n_surveyed}개** / {len(meta.column_names):,}개")
        L.append(f"- 주 문항 후보 **{n_cand}개** (A/B {n_ab}개)")
        L.append(f"- 조건부 문항 **{n_cond}개** (스킵 비해당 {args.max_iap:.0%} 초과)")
        L.append(f"- 셀 구성 **{' x '.join(chosen)}**\n")

        L.append("### 셀 구성별 표본 크기\n")
        L.append("| 셀 구성 | 셀 수 | 최소 n | 중앙 n | n>=20 |\n|---|---|---|---|---|")
        for _, r in cellcheck.iterrows():
            L.append(f"| {r['cells']} | {r['n_cells']} | {r['min_n']} "
                     f"| {r['median_n']:.0f} | {r['pct_ge20']}% |")
        L.append("")

        if base is not None and not base.empty:
            L.append("### 개인 예측 정확도의 상한 (셀 최빈응답)\n")
            L.append(f"- 전체 최빈 정확도 중앙값 **{base['marginal_modal'].median():.3f}**")
            L.append(f"- 셀 최빈 정확도 중앙값 **{base['cell_modal'].median():.3f}**")
            L.append(f"- 인구통계로 얻는 순이익 중앙값 **{base['gain'].median():+.3f}**\n")
            L.append("> LLM 정확도는 전체 최빈을 넘어야 의미가 있고, 셀 최빈을 넘으면 "
                     "인구통계 이상의 무언가를 쓰고 있다는 뜻입니다. "
                     "eta2와 달리 명목형 문항에도 정의됩니다.\n")
            L.append("#### 인구통계 이득 상위 10\n")
            L.append("| 문항 | 라벨 | 전체최빈 | 셀최빈 | 이득 |\n|---|---|---|---|---|")
            for _, r in base.head(10).iterrows():
                L.append(f"| `{r['item']}` | {str(r['label'])[:35]} "
                         f"| {r['marginal_modal']} | {r['cell_modal']} | {r['gain']:+.4f} |")
            L.append("")

        if eta is not None and not eta.empty:
            L.append("### eta2 (문항 선정의 층화 축)\n")
            L.append(f"- 대상 **{len(eta)}문항** / 중앙값 **{eta['eta2'].median():.3f}**\n")
            L.append("> 재현도와의 관계는 검증할 가설(H3, H4)이며 확립된 사실이 아닙니다.\n")
            for title, part in [("상위 10", eta.head(10)), ("하위 10", eta.tail(10).iloc[::-1])]:
                L.append(f"#### {title}\n| 문항 | 라벨 | eta2 | 무응답률 |\n|---|---|---|---|")
                for _, r in part.iterrows():
                    L.append(f"| `{r['item']}` | {str(r['label'])[:35]} | {r['eta2']} "
                             f"| {r['dk_rate']*100:.1f}% |")
                L.append("")

        L.append("### 조건부 문항 (스킵 준수율 측정 대상)\n")
        L.append("> 비해당인 페르소나에게 물었을 때 '해당 없음'이라 답하는지를 잽니다. "
                 "실제 조사에서는 조사원이 건너뛰므로 준수율이 정의상 100%입니다. "
                 "비해당자 수가 많을수록 측정 표본이 커집니다.\n")
        if n_cond:
            top = pd.read_csv(outdir / "conditional_items.csv").head(15)
            L.append("| 문항 | 라벨 | 비해당률 | 응답자 | 비해당자 |\n|---|---|---|---|---|")
            for _, r in top.iterrows():
                L.append(f"| `{r['var']}` | {str(r['label'])[:35]} "
                         f"| {r['iap_rate']*100:.0f}% | {int(r['n_eligible'])} "
                         f"| {int(r['n_inapplicable'])} |")
            L.append("")

    (outdir / "diagnostics.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n완료 -> {outdir.resolve()}")


if __name__ == "__main__":
    main()