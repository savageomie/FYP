"""
Compute enrichment across all 15 disorders in VulnMap and save to processed data.
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from vulnmap.enrichment import compute_enrichment
from vulnmap.nulls import create_null_model
from vulnmap.damage import DK_ALL_REGIONS
from vulnmap.utils import get_project_root, load_params, logger

def main():
    root = get_project_root()
    expr_path = root / "data" / "interim" / "expression_desikan_killiany.parquet"
    expr = pd.read_parquet(expr_path)
    if not isinstance(expr.index[0], str):
        expr = expr.iloc[:len(DK_ALL_REGIONS)].copy()
        expr.index = DK_ALL_REGIONS

    gs_path = root / "data" / "interim" / "harmonized_genesets.json"
    gs_data = json.load(open(gs_path))

    params = load_params()
    null_model_names = ["naive", "matched", "spatial_coexpr"]
    n_perm = 200
    seed = 42

    logger.info("Pre-instantiating null models...")
    null_models = {}
    for null_name in null_model_names:
        null_models[null_name] = create_null_model(
            name=null_name,
            expression_df=expr,
            n_perm=n_perm,
            seed=seed,
            params=params,
        )

    all_results = []
    t_start = time.time()

    for gs_key, gene_list in gs_data.items():
        if isinstance(gs_key, str) and "__" in gs_key:
            disorder, geneset_def = gs_key.split("__", 1)
        else:
            disorder, geneset_def = str(gs_key), "default"

        if len(gene_list) < 5:
            logger.warning(f"Skipping {disorder}/{geneset_def} (too few genes: {len(gene_list)})")
            continue

        logger.info(f"Processing {disorder} [{geneset_def}] ({len(gene_list)} genes)...")

        for null_name in null_model_names:
            try:
                res = compute_enrichment(
                    expression_df=expr,
                    gene_set=gene_list,
                    null_model=null_models[null_name],
                    score_method="mean_zscore",
                    fdr_alpha=0.05,
                )
                res.insert(0, "disorder", disorder)
                res.insert(1, "geneset_def", geneset_def)
                res.insert(2, "null_model", null_name)
                all_results.append(res)
            except Exception as e:
                logger.error(f"Failed {disorder}/{geneset_def}/{null_name}: {e}")

    full_df = pd.concat(all_results, ignore_index=True)
    out_dir = root / "data" / "processed"
    full_df.to_parquet(out_dir / "enrichment_results.parquet", index=False)
    full_df.to_csv(out_dir / "enrichment_results.csv", index=False)
    elapsed = round(time.time() - t_start, 2)
    print(f"SUCCESS: Completed 15 disorders enrichment in {elapsed}s. Total rows: {len(full_df)}")

if __name__ == "__main__":
    main()
