r"""
KGSS 코드북 PDF 설문 원문 추출 + A/B 분할표본 효과 분석

.sav의 변수 라벨은 축약본이다. 실제 응답자가 들은 문장은 코드북에만 있다.
페르소나 프롬프트에는 반드시 코드북 원문을 써야 한다.

A/B 분할표본:
    2016년부터 홀수번호 가구는 A형, 짝수번호 가구는 B형 설문지를 받았다(조사실험).
    무작위 배정이므로 두 그룹의 인구 구성은 통계적으로 동일하고,
    응답 차이는 설문지 차이 때문이라고 인과적으로 말할 수 있다.

    조작 유형은 셋이다.
      - 선택지 순서 역전: PROUD* 10개, NUKPLT10, LAWHARSH, ELEFRAUD
      - 질문 문구:        SAMPTHOU("무작위" 유무)
      - 목록 항목 추가:   RUNELECT(페미니스트), DISCRNUM(여성)

    순서 역전 쌍은 B를 A의 코딩으로 재정렬한 뒤 두 분포를 비교해
    '인간에게서 나타나는 응답 순서 효과'의 크기를 산출한다.
    이 값이 LLM 검증의 기준선이 된다.

주의: 코드북 표의 백분율은 가중치(FINALWT)가 적용된 값이다.
      우리 계산(비가중)과 어긋나면 이것을 먼저 의심할 것.

사용법:
    uv pip install pdfplumber
    uv run kgss_codebook.py --pdf "2003-2025_KGSS_Codebook_v5.pdf" ^
        --inv .\inventory --out .\selection
"""

import argparse
import io
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

STRUCTURAL = {-8.0, -1.0}
FORM_PAT = re.compile(r"\(\s*([AB])\s*형\s*\)")
BLOCK_PAT = re.compile(
    r"표\s*(\d+)\s*([^\n]*?)\n(?:\[설문\s*문항\](.*?))?\[변수명\]\s*(\S+)", re.S
)


def norm(s: str) -> str:
    return " ".join((s or "").split())


def extract_with_pdftotext(pdf: str, out: Path) -> bool:
    exe = shutil.which("pdftotext")
    if not exe:
        return False
    print("      poppler 발견 -> pdftotext -raw")
    subprocess.run([exe, "-raw", pdf, str(out)], check=True)
    return True


def extract_with_pdfplumber(pdf: str, out: Path) -> None:
    try:
        import pdfplumber
    except ImportError:
        sys.exit("uv pip install pdfplumber 후 재실행하세요.")
    print("      poppler 없음 -> pdfplumber (약 10분, 1회만)")
    chunks = []
    with pdfplumber.open(pdf) as doc:
        total = len(doc.pages)
        for i, page in enumerate(doc.pages, 1):
            chunks.append(page.extract_text(x_tolerance=1.5, y_tolerance=3) or "")
            if i % 50 == 0 or i == total:
                print(f"      {i:>4}/{total} 쪽", flush=True)
    out.write_text("\n".join(chunks), encoding="utf-8")


def ensure_txt(pdf: str | None, txt: Path) -> Path:
    if txt.exists() and txt.stat().st_size > 1000:
        print(f"[0] 캐시 사용: {txt}")
        return txt
    if not pdf or not Path(pdf).exists():
        sys.exit(f"{txt}도 --pdf도 없습니다.")
    print(f"[0] 텍스트 추출 ({pdf})")
    if not extract_with_pdftotext(pdf, txt):
        extract_with_pdfplumber(pdf, txt)
    return txt


def resolve_name(cb_var: str, columns: set[str]) -> str | None:
    """코드북 변수명 -> 실제 .sav 컬럼명. SAMPTHOUB -> SAMPTHOUB23"""
    if cb_var in columns:
        return cb_var
    for c in columns:
        if c.startswith(cb_var) and re.fullmatch(r"\d{2}", c[len(cb_var):]):
            return c
    base = re.sub(r"\d{2}$", "", cb_var)
    if base != cb_var and base in columns:
        return base
    return None


def fmt_options(vl: dict) -> str:
    parts = []
    for k, lab in sorted(vl.items(), key=lambda kv: float(kv[0])):
        c = float(k)
        if c in STRUCTURAL:
            continue
        parts.append(f"{int(c) if c == int(c) else c}={lab}")
    return " | ".join(parts)


def substantive(vl: dict, dk: list, inap: list) -> dict:
    drop = STRUCTURAL | set(dk) | set(inap)
    return {float(k): norm(str(v)) for k, v in vl.items() if float(k) not in drop}


def reversal_map(a: dict, b: dict) -> dict | None:
    """B의 선택지가 A의 정확한 역순이면 B코드 -> A코드 매핑을 반환."""
    ka, kb = sorted(a), sorted(b)
    if len(ka) != len(kb) or len(ka) < 2:
        return None
    la, lb = [a[k] for k in ka], [b[k] for k in kb]
    if la == lb or lb != la[::-1]:
        return None
    return {kb[i]: ka[len(ka) - 1 - i] for i in range(len(ka))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--txt", default="codebook_raw.txt")
    ap.add_argument("--inv", default="./inventory")
    ap.add_argument("--out", default="./selection")
    args = ap.parse_args()

    inv, outdir = Path(args.inv), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    txt = ensure_txt(args.pdf, Path(args.txt))

    print("[1/5] 코드북 파싱")
    raw = txt.read_text(encoding="utf-8", errors="replace")
    recs = {}
    for m in BLOCK_PAT.finditer(raw):
        recs[m.group(4)] = {"표제목": norm(m.group(2)), "설문원문": norm(m.group(3))}
    print(f"      {len(recs):,}개 변수 원문 추출")

    print("[2/5] 데이터 로드 및 이름 대응")
    df = pd.read_parquet(inv / "kgss_clean.parquet")
    vt = pd.read_csv(inv / "variables.csv")
    cols = set(df.columns)

    name_map = {v: resolve_name(v, cols) for v in recs}
    renamed = {v: c for v, c in name_map.items() if c and c != v}
    print(f"      이름 변경 {len(renamed):,}개 / 대응 실패 "
          f"{sum(1 for c in name_map.values() if c is None):,}개")

    rev = {}
    for v, c in name_map.items():
        if c:
            rev.setdefault(c, v)
    vt["설문원문"] = vt["var"].map(lambda c: recs.get(rev.get(c, c), {}).get("설문원문", ""))
    vt["표제목"] = vt["var"].map(lambda c: recs.get(rev.get(c, c), {}).get("표제목", ""))
    vt["선택지"] = vt["value_labels"].map(
        lambda s: fmt_options(json.loads(s)) if isinstance(s, str) else ""
    )
    print(f"      원문 없음 {int((vt['설문원문'] == '').sum()):,}개 / {len(vt):,}개")
    keep = [c for c in ["var", "label", "표제목", "설문원문", "선택지",
                        "measure", "ordinal_guess", "dk", "inap"] if c in vt.columns]
    vt[keep].to_csv(outdir / "variables_worded.csv", index=False, encoding="utf-8-sig")

    print("[3/5] SAMPLEAB 실제 코드 확인")
    L = ["# 설문 원문 및 A/B 분할표본\n"]
    L.append("> 코드북 표의 백분율은 가중치(FINALWT)가 적용된 값입니다. "
             "아래 인원수는 비가중 실측치이므로 코드북과 소수점 차이가 납니다.\n")
    if "SAMPLEAB" in df.columns:
        ct = (df[df["SAMPLEAB"].notna()].groupby(["YEAR", "SAMPLEAB"])
              .size().unstack(fill_value=0))
        L.append("## SAMPLEAB 실제 값 분포\n")
        L.append("| 연도 | " + " | ".join(f"값 {c:g}" for c in ct.columns) + " |")
        L.append("|" + "---|" * (len(ct.columns) + 1))
        for yr, r in ct.iterrows():
            L.append(f"| {int(yr)} | " + " | ".join(f"{int(x):,}" for x in r) + " |")
        L.append("")
        print(f"      SAMPLEAB 실제 값: {sorted(ct.columns.tolist())}")

    print("[4/5] A/B 쌍 분석")
    info = vt.set_index("var")
    years = sorted(df["YEAR"].unique())

    forms = {}
    for v, r in recs.items():
        m = FORM_PAT.search(r["표제목"])
        if not m:
            continue
        stem = re.sub(r"\d{2}$", "", v)
        stem = stem[:-1] if stem and stem[-1] in "AB" else stem
        forms.setdefault(stem, {})[m.group(1)] = v

    rows = []
    for stem, d in sorted(forms.items()):
        if not ("A" in d and "B" in d):
            continue
        ca, cb = name_map.get(d["A"]), name_map.get(d["B"])
        if not ca or not cb:
            continue
        row = {"쌍": stem, "A변수": ca, "B변수": cb}
        opts = {}
        for side, cv, cbv in (("A", ca, d["A"]), ("B", cb, d["B"])):
            row[f"{side}원문"] = recs[cbv]["설문원문"]
            vl = json.loads(info.at[cv, "value_labels"]) if cv in info.index else {}
            dk = json.loads(info.at[cv, "dk"]) if cv in info.index else []
            inap = json.loads(info.at[cv, "inap"]) if cv in info.index else []
            opts[side] = substantive(vl, dk, inap)
            row[f"{side}선택지"] = fmt_options(vl)
            yrs = [int(y) for y in years if df.loc[df["YEAR"] == y, cv].notna().any()]
            row[f"{side}연도"] = ", ".join(map(str, yrs))
            row[f"{side}_n"] = int(df[cv].notna().sum())

        row["문구다름"] = row["A원문"] != row["B원문"]
        row["선택지다름"] = row["A선택지"] != row["B선택지"]

        rmap = reversal_map(opts["A"], opts["B"])
        row["순서역전"] = rmap is not None
        for k in ("효과_연도", "효과_평균차", "효과_TV거리", "효과_분포A", "효과_분포B"):
            row[k] = np.nan
        if rmap:
            common = set(row["A연도"].split(", ")) & set(row["B연도"].split(", "))
            common.discard("")
            if common:
                y = int(sorted(common)[-1])
                g = df[df["YEAR"] == y]
                codes = sorted(opts["A"])
                sa = g[ca][g[ca].isin(codes)]
                sb = g[cb].map(rmap).dropna()
                sb = sb[sb.isin(codes)]
                if len(sa) > 30 and len(sb) > 30:
                    pa = np.array([(sa == c).mean() for c in codes])
                    pb = np.array([(sb == c).mean() for c in codes])
                    row["효과_연도"] = y
                    row["효과_평균차"] = round(float(sa.mean() - sb.mean()), 4)
                    row["효과_TV거리"] = round(float(np.abs(pa - pb).sum() / 2), 4)
                    row["효과_분포A"] = " ".join(f"{x:.3f}" for x in pa)
                    row["효과_분포B"] = " ".join(f"{x:.3f}" for x in pb)
        rows.append(row)

    abdf = pd.DataFrame(rows)
    abdf.to_csv(outdir / "ab_pairs.csv", index=False, encoding="utf-8-sig")
    n_rev = int(abdf["순서역전"].sum()) if not abdf.empty else 0
    print(f"      A/B 쌍 {len(abdf)}개 / 순서 역전 {n_rev}개")

    print("[5/5] 보고서 작성")
    if not abdf.empty:
        L.append(f"## A/B 쌍 {len(abdf)}개 (순서 역전 {n_rev}개)\n")
        L.append("| 쌍 | 조작 | A 회차 | B 회차 | A n | B n | TV거리 | 평균차 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for _, r in abdf.iterrows():
            what = "순서역전" if r["순서역전"] else (
                "문구" if r["문구다름"] else ("선택지" if r["선택지다름"] else "-"))
            tv = "" if pd.isna(r["효과_TV거리"]) else f"{r['효과_TV거리']:.3f}"
            md = "" if pd.isna(r["효과_평균차"]) else f"{r['효과_평균차']:+.4f}"
            L.append(f"| `{r['쌍']}` | {what} | {r['A연도']} | {r['B연도']} "
                     f"| {r['A_n']:,} | {r['B_n']:,} | {tv} | {md} |")
        L.append("\n> TV거리는 선택지 순서를 뒤집었을 때 인간 응답 분포가 실제로 이동한 "
                 "확률질량의 비율입니다. 0.05면 약 5%의 응답자가 순서 때문에 다르게 "
                 "답했다는 뜻입니다. LLM이 이 크기와 방향을 재현하는지가 검증 대상입니다.\n")

        # 순서 효과 방향 일관성
        rev_df = abdf[abdf["순서역전"] & abdf["효과_평균차"].notna()]
        if len(rev_df) >= 3:
            pos = int((rev_df["효과_평균차"] > 0).sum())
            L.append("### 응답 순서 효과의 방향 일관성\n")
            L.append(f"- 순서 역전 쌍 {len(rev_df)}개 중 **{pos}개가 양의 평균차**")
            L.append(f"- TV거리 중앙값 **{rev_df['효과_TV거리'].median():.3f}** "
                     f"(범위 {rev_df['효과_TV거리'].min():.3f}~"
                     f"{rev_df['효과_TV거리'].max():.3f})\n")
            L.append("> 평균차가 한 방향으로 몰리면 우연이 아닌 체계적 순서 효과입니다. "
                     "대면 면접에서는 뒤에 제시된 선택지를 고르는 최신 효과(recency)가 "
                     "나타나는 것으로 알려져 있습니다.\n")

        for _, r in abdf.iterrows():
            L.append(f"### `{r['쌍']}`\n")
            L.append(f"**A형** `{r['A변수']}` — {r['A연도']} (n={r['A_n']:,})\n")
            L.append(f"- 문항: {r['A원문']}")
            L.append(f"- 선택지: {r['A선택지']}\n")
            L.append(f"**B형** `{r['B변수']}` — {r['B연도']} (n={r['B_n']:,})\n")
            L.append(f"- 문항: {r['B원문']}")
            L.append(f"- 선택지: {r['B선택지']}\n")
            if r["순서역전"] and not pd.isna(r["효과_TV거리"]):
                L.append(f"**선택지 순서 역전 실험** ({int(r['효과_연도'])}년)\n")
                L.append(f"- A형 분포: {r['효과_분포A']}")
                L.append(f"- B형 분포(A 코딩으로 정렬): {r['효과_분포B']}")
                L.append(f"- TV거리 **{r['효과_TV거리']:.3f}** / "
                         f"평균차 {r['효과_평균차']:+.4f}\n")
            else:
                what = [w for w, ok in [("질문 문구", r["문구다름"]),
                                        ("응답 선택지", r["선택지다름"])] if ok]
                L.append(f"→ 조작: **{', '.join(what) if what else '동일'}**\n")

    (outdir / "codebook_ab.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n완료 -> {outdir.resolve()}")


if __name__ == "__main__":
    main()