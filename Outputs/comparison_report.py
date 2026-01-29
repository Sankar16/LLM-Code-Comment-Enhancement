# === Comparison tables for 3 models (3 examples per category) ===
# Set 1 = Llama-3, Set 2 = Phi-2, Set 3 = Phi-3-mini
import json, os, random, pandas as pd
from textwrap import shorten

# Change if your files are elsewhere
BASE = "Outputs"  # e.g., "/content" in Colab if that’s where your files are

categories = ["what", "why", "how-to-use", "how-it-is-done", "property", "others"]
file_patterns = {
    "Set 1 (Llama-3)": "ans_{cat}.json",
    "Set 2 (Phi-2)": "ans_{cat}-2.json",
    "Set 3 (Phi-3-mini)": "ans_{cat}-3.json",
}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def coerce_record(rec):
    # Normalize common keys across different exporters
    id_ = rec.get("id") or rec.get("sample_id") or rec.get("idx")
    cat = rec.get("category") or rec.get("label") or rec.get("type") or rec.get("class")
    raw = rec.get("raw_code") or rec.get("code") or rec.get("source_code")
    gold = (rec.get("gold_comment") or rec.get("reference") or
            rec.get("target_comment") or rec.get("true_comment"))
    # If the JSON kept only one "comment" and also a generated field, assume "comment" is gold
    if gold is None and "gold" in rec: gold = rec["gold"]
    if gold is None and "comment" in rec: gold = rec["comment"]
    gen = (rec.get("generated_comment") or rec.get("prediction") or
           rec.get("output") or rec.get("model_comment") or rec.get("gen_comment"))
    return str(id_) if id_ is not None else None, cat, raw, gold, (gen or "")

def merge_category(cat):
    merged = {}  # id -> row
    for model_name, pattern in file_patterns.items():
        path = os.path.join(BASE, pattern.format(cat=cat))
        if not os.path.exists(path):
            print(f"⚠️ Missing: {path}")
            continue
        data = load_json(path)
        for rec in data:
            id_, c, raw, gold, gen = coerce_record(rec)
            if id_ is None: 
                continue
            row = merged.setdefault(id_, {
                "id": id_,
                "category": cat,
                "raw_code": raw,
                "gold_comment": gold,
                "Set 1 (Llama-3)": "",
                "Set 2 (Phi-2)": "",
                "Set 3 (Phi-3-mini)": "",
            })
            row[model_name] = gen
            # backfill base fields if missing
            if row["raw_code"] is None and raw is not None: row["raw_code"] = raw
            if row["gold_comment"] is None and gold is not None: row["gold_comment"] = gold
    return merged

def build_tables(sample_per_cat=3, seed=7):
    random.seed(seed)
    all_tables = {}
    os.makedirs(os.path.join(BASE, "comparison_tables"), exist_ok=True)
    excel_path = os.path.join(BASE, "comparison_tables", "model_comment_comparison.xlsx")
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        for cat in categories:
            merged = merge_category(cat)
            if not merged:
                continue
            df = pd.DataFrame(list(merged.values()))
            # preview column for slides
            df.insert(2, "raw_code_preview", df["raw_code"].apply(
                lambda s: shorten(s.replace("\n", " "), width=140, placeholder=" …") if isinstance(s, str) else s
            ))
            df = df[["id", "category", "raw_code_preview", "gold_comment",
                     "Set 1 (Llama-3)", "Set 2 (Phi-2)", "Set 3 (Phi-3-mini)"]]
            df = df.sort_values(by="id", key=lambda x: x.astype(str))
            # Save full CSV for archive
            full_csv = os.path.join(BASE, "comparison_tables", f"comparison_{cat}_full.csv")
            df.to_csv(full_csv, index=False)
            # Pick up to 3 rows deterministically
            rows = df.to_dict("records")
            if len(rows) > sample_per_cat:
                idxs = random.sample(range(len(rows)), sample_per_cat)
                rows = [rows[i] for i in sorted(idxs)]
            small = pd.DataFrame(rows)
            # Save small CSV for quick sharing
            small_csv = os.path.join(BASE, "comparison_tables", f"comparison_{cat}_sample.csv")
            small.to_csv(small_csv, index=False)
            # Also write each sheet in Excel (sample view for meetings)
            small.to_excel(writer, sheet_name=cat[:31], index=False)
            all_tables[cat] = small
    return all_tables, excel_path

tables, excel_path = build_tables(sample_per_cat=3, seed=11)

    
print(f"\n✅ Saved per-category CSVs and Excel workbook here:\n- Excel: {excel_path}\n- CSVs: {os.path.join(BASE, 'comparison_tables')}")