#!/usr/bin/env python3
"""Build annotated h5ad files from downloaded HTAN CalicoST Visium matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import KDTree


POSITION_COLUMNS = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
]

PASTE_COORDINATE_FILES = {
    "HT112C1-U1_ST_Bn1": "ht112u1.csv",
    "HT112C1-U2_ST_Bn1": "ht112u2.csv",
    "HT268B1-Th1K3Fc2U1Z1Bs1": "Th1K3U1.csv",
    "HT268B1-Th1H3Fc2U12Z1Bs1": "Th1H3U12.csv",
    "HT268B1-Th1H3Fc2U2Z1Bs1": "Th1H3U2.csv",
    "HT268B1-Th1H3Fc2U22Z1Bs1": "Th1H3U22.csv",
    "HT268B1-Th1H3Fc2U32Z1Bs1": "Th1H3U32.csv",
}


def _matrix_prefix(raw_dir: Path) -> str:
    matrices = list(raw_dir.glob("*-matrix.mtx.gz"))
    if len(matrices) != 1:
        raise ValueError(f"Expected one matrix in {raw_dir}, found {len(matrices)}")
    return matrices[0].name[: -len("matrix.mtx.gz")]


def _load_sample_table(deposit_dir: Path) -> pd.DataFrame:
    samples = pd.read_csv(deposit_dir / "samplelist.tsv", sep="\t")
    return samples.set_index("slice_ids", drop=False)


def _median_nn(coords: np.ndarray) -> float | None:
    coords = np.asarray(coords, dtype=np.float64)
    coords = coords[np.isfinite(coords).all(axis=1)]
    if coords.shape[0] < 2:
        return None
    return float(np.median(KDTree(coords).query(coords, k=2)[0][:, 1]))


def _visium_um_per_fullres_pixel(scalefactors: dict) -> float | None:
    try:
        spot_px = float(scalefactors.get("spot_diameter_fullres"))
    except (TypeError, ValueError):
        return None
    if spot_px <= 0:
        return None
    return 55.0 / spot_px


def _estimate_paste_um_per_unit(
    *,
    spaceranger_coords: np.ndarray | None,
    paste_coords: np.ndarray,
    scalefactors: dict,
) -> float | None:
    px_um = _visium_um_per_fullres_pixel(scalefactors)
    if px_um is None:
        return None

    if spaceranger_coords is not None:
        sr_nn = _median_nn(spaceranger_coords)
        paste_nn = _median_nn(paste_coords)
        if sr_nn is not None and paste_nn is not None and paste_nn > 0:
            return sr_nn * px_um / paste_nn

    # Some PASTE files are already in a fullres-like coordinate frame.
    return px_um


def _add_spatial_data(
    adata,
    slice_id: str,
    patient: str,
    deposit_dir: Path,
) -> str:
    spatial_dir = deposit_dir / "Spatial_coordinates" / slice_id
    positions_path = spatial_dir / "tissue_positions_list.csv"
    with (spatial_dir / "scalefactors_json.json").open() as handle:
        scalefactors = json.load(handle)

    spaceranger_positions = None
    spaceranger_coords = None
    if positions_path.exists():
        spaceranger_positions = pd.read_csv(
            positions_path,
            header=None,
            names=POSITION_COLUMNS,
            index_col="barcode",
        )
        spaceranger_positions = spaceranger_positions.reindex(adata.obs_names)
        missing = spaceranger_positions["pxl_row_in_fullres"].isna()
        if missing.any():
            raise ValueError(
                f"{slice_id}: {int(missing.sum())} expression barcodes lack spatial coordinates"
            )
        spaceranger_coords = spaceranger_positions[
            ["pxl_col_in_fullres", "pxl_row_in_fullres"]
        ].to_numpy(dtype=np.float32)

    paste_name = PASTE_COORDINATE_FILES.get(slice_id)
    paste_path = (
        deposit_dir / "PASTE_alignments" / patient / paste_name
        if paste_name is not None
        else None
    )
    use_paste = paste_path is not None and paste_path.exists()
    if use_paste:
        paste = pd.read_csv(paste_path, index_col="barcode")
        if not {"x", "y"}.issubset(paste.columns):
            raise ValueError(f"{paste_path} must contain x and y columns")
        paste = paste.reindex(adata.obs_names)
        missing = paste[["x", "y"]].isna().any(axis=1)
        if spaceranger_positions is not None:
            tissue = spaceranger_positions["in_tissue"].to_numpy(dtype=bool)
            bad_missing = missing.to_numpy() & tissue
        else:
            bad_missing = np.zeros(adata.n_obs, dtype=bool)
        if np.any(bad_missing):
            raise ValueError(
                f"{slice_id}: {int(np.sum(bad_missing))} tissue barcodes lack PASTE coordinates"
            )

        if spaceranger_positions is not None:
            positions = spaceranger_positions.copy()
            adata.obsm["spatial_spaceranger_fullres"] = spaceranger_coords
        else:
            positions = pd.DataFrame(index=adata.obs_names)
            positions["in_tissue"] = 1
            positions["array_row"] = np.nan
            positions["array_col"] = np.nan
            positions["pxl_row_in_fullres"] = np.nan
            positions["pxl_col_in_fullres"] = np.nan

        positions["paste_x"] = paste["x"].to_numpy(dtype=np.float32)
        positions["paste_y"] = paste["y"].to_numpy(dtype=np.float32)
        paste_coords = paste[["x", "y"]].to_numpy(dtype=np.float32)
        adata.obsm["spatial"] = paste_coords
        coordinate_source = f"CalicoST PASTE alignment: {patient}/{paste_name}"
        paste_um_per_unit = _estimate_paste_um_per_unit(
            spaceranger_coords=spaceranger_coords,
            paste_coords=paste_coords,
            scalefactors=scalefactors,
        )
    else:
        if spaceranger_positions is None:
            raise FileNotFoundError(f"No spatial coordinates available for {slice_id}")
        positions = spaceranger_positions
        adata.obsm["spatial"] = spaceranger_coords
        coordinate_source = "Space Ranger tissue_positions_list.csv"
        paste_um_per_unit = None

    for column in POSITION_COLUMNS[1:]:
        adata.obs[column] = positions[column].to_numpy()
    if "paste_x" in positions.columns:
        adata.obs["paste_x"] = positions["paste_x"].to_numpy(dtype=np.float32)
        adata.obs["paste_y"] = positions["paste_y"].to_numpy(dtype=np.float32)

    metadata = {
        "source": "HTAN WUSTL 10X Visium",
        "coordinate_source": coordinate_source,
    }
    if use_paste:
        metadata["original_coordinate_source"] = (
            "Space Ranger tissue_positions_list.csv"
            if spaceranger_positions is not None
            else "not available in deposit"
        )
        metadata["paste_file"] = f"PASTE_alignments/{patient}/{paste_name}"
        if paste_um_per_unit is not None:
            metadata["coordinate_um_per_unit"] = float(paste_um_per_unit)

    adata.uns["spatial"] = {
        slice_id: {
            "scalefactors": scalefactors,
            "metadata": metadata,
        }
    }
    return coordinate_source


def _add_calicost_annotations(
    adata,
    sample: pd.Series,
    deposit_dir: Path,
) -> None:
    patient = sample["patient"]
    results_dir = deposit_dir / "CalicoST_results" / patient

    labels = pd.read_csv(results_dir / "clone_labels.tsv", sep="\t")
    labels = labels.set_index("BARCODES")
    suffix = str(sample["slice_numbering"])
    candidates = pd.Index([f"{barcode}_{suffix}" for barcode in adata.obs_names])

    direct_hits = adata.obs_names.isin(labels.index).sum()
    suffixed_hits = candidates.isin(labels.index).sum()
    lookup = candidates if suffixed_hits > direct_hits else adata.obs_names
    matched = labels.reindex(lookup)

    adata.obs["calicost_clone_label"] = matched["clone_label"].to_numpy()
    adata.obs["calicost_tumor_proportion"] = matched[
        "tumor_proportion"
    ].to_numpy(dtype=np.float32)

    cnv = pd.read_csv(results_dir / "cnv_genelevel.tsv", sep="\t")
    cnv = cnv.drop_duplicates("gene").set_index("gene")
    gene_symbols = pd.Index(adata.var["gene_symbol"].astype(str))
    aligned = cnv.reindex(gene_symbols)
    adata.varm["calicost_allele_cnv"] = aligned.to_numpy(dtype=np.float32)
    adata.uns["calicost_allele_cnv_columns"] = np.asarray(
        cnv.columns.astype(str), dtype=str
    )
    adata.uns["calicost"] = {
        "patient": str(patient),
        "cancer_type": str(sample["cancer_type"]),
        "slice_numbering": suffix,
        "clone_label_column": "calicost_clone_label",
        "tumor_proportion_column": "calicost_tumor_proportion",
        "allele_cnv_varm_key": "calicost_allele_cnv",
        "allele_cnv_columns_uns_key": "calicost_allele_cnv_columns",
        "normal_clone_copy_number": "Clone label 0 represents normal/diploid spots.",
    }


def build_one(
    slice_id: str,
    sample: pd.Series,
    raw_root: Path,
    output_root: Path,
    deposit_dir: Path,
    force: bool,
) -> Path:
    output_path = output_root / f"{slice_id}.h5ad"
    if output_path.exists() and not force:
        print(f"[skip] {output_path}")
        return output_path

    raw_dir = raw_root / slice_id
    prefix = _matrix_prefix(raw_dir)
    adata = sc.read_10x_mtx(
        raw_dir,
        var_names="gene_symbols",
        make_unique=False,
        cache=False,
        prefix=prefix,
        gex_only=True,
    )
    adata.var["gene_symbol"] = adata.var_names.astype(str)
    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()

    coordinate_source = _add_spatial_data(
        adata,
        slice_id,
        str(sample["patient"]),
        deposit_dir,
    )
    _add_calicost_annotations(adata, sample, deposit_dir)

    if coordinate_source.startswith("CalicoST PASTE"):
        spatial_missing = ~np.isfinite(np.asarray(adata.obsm["spatial"], dtype=np.float32)).all(axis=1)
        labeled = adata.obs["calicost_clone_label"].notna().to_numpy()
        bad_missing = spatial_missing & labeled
        if np.any(bad_missing):
            raise ValueError(
                f"{slice_id}: {int(np.sum(bad_missing))} CalicoST-labeled spots lack PASTE coordinates"
            )

    # The deposited Level-3 matrices contain the full 4,992-spot Visium array.
    # Restrict outputs to tissue spots. For the four HT268B1 slices whose raw
    # Space Ranger positions were not deposited, CalicoST clone labels define
    # the assayed tissue spots.
    if coordinate_source.startswith("Space Ranger"):
        tissue_mask = adata.obs["in_tissue"].to_numpy(dtype=bool)
    else:
        tissue_mask = adata.obs["calicost_clone_label"].notna().to_numpy()
        adata.obs["in_tissue"] = tissue_mask.astype(np.int8)
    adata = adata[tissue_mask].copy()

    adata.obs["slice_id"] = slice_id
    adata.obs["patient"] = str(sample["patient"])
    adata.obs["cancer_type"] = str(sample["cancer_type"])
    adata.uns["source"] = {
        "expression": "HTAN WUSTL Level 3 10X Visium",
        "calicost": "Zenodo 10.5281/zenodo.14175627",
    }

    output_root.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path, compression="gzip")
    matched = int(adata.obs["calicost_clone_label"].notna().sum())
    print(
        f"[write] {output_path} "
        f"({adata.n_obs} spots x {adata.n_vars} genes; {matched} clone labels)"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/h5ad/calicost"),
        help="CalicoST data directory",
    )
    parser.add_argument("--slice", action="append", dest="slices")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.root
    deposit_dir = root / "CalicoST_deposit_data"
    samples = _load_sample_table(deposit_dir)
    samples = samples[samples["cancer_type"] != "Prostate"]
    if args.slices:
        missing = sorted(set(args.slices) - set(samples.index))
        if missing:
            raise ValueError(f"Unknown or unavailable slices: {missing}")
        samples = samples.loc[args.slices]

    for slice_id, sample in samples.iterrows():
        build_one(
            slice_id=slice_id,
            sample=sample,
            raw_root=root / "raw",
            output_root=root,
            deposit_dir=deposit_dir,
            force=args.force,
        )


if __name__ == "__main__":
    main()
