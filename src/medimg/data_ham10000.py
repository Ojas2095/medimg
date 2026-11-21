"""Loader and preprocessor for HAM10000 + ISIC 2018 Task 1.

1. Aligns HAM10000 metadata (lesion_id, image_id, dx, etc.) with image files.
2. Binary target: melanoma ('mel', label=1) vs other ('nv', 'bkl', 'bcc', 'akiec', 'vasc', 'df', label=0).
   Melanoma is the deadliest skin malignancy; prioritizing high-risk melanoma reduces diagnostic delay.
   Retains the 7-class diagnosis in 'dx_7class' for granular evaluation.
3. Aligns ISIC 2018 Task 1 segmentation masks (or creates synthetic/empty zero-masks if an image has no mask).
4. Derives bounding boxes [x1, y1, x2, y2] from connected components of binary lesion masks via scipy.ndimage.
5. Emits standard metadata.csv (image_id, patient_id, label, image_path, mask_path, boxes_json, dx_7class).
"""
import argparse, json, os, glob
import numpy as np, pandas as pd
from PIL import Image
from scipy.ndimage import label as nd_label

DX_7CLASSES = ["mel", "nv", "bkl", "bcc", "akiec", "vasc", "df"]

def mask_to_boxes(mask_np):
    """Derive bounding boxes [x1, y1, x2, y2] from connected components in binary mask."""
    binary = (mask_np > 127).astype(np.uint8)
    if binary.sum() == 0:
        return []
    labeled, num_features = nd_label(binary)
    boxes = []
    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) > 0 and len(ys) > 0:
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
    return boxes

def locate_metadata_file(raw_dir):
    candidates = [
        os.path.join(raw_dir, "HAM10000_metadata_active.csv"),
        os.path.join(raw_dir, "HAM10000_metadata.csv"),
        os.path.join(raw_dir, "metadata.csv"),
        os.path.join(raw_dir, "HAM10000_metadata"),
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            return c
    found = glob.glob(os.path.join(raw_dir, "*metadata*.csv"))
    if found:
        return found[0]
    return None

def find_image_file(raw_dir, image_id):
    extensions = [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]
    subdirs = ["", "images", "HAM10000_images_part_1", "HAM10000_images_part_2", "ISIC2018_Task3_Training_Input"]
    for sub in subdirs:
        for ext in extensions:
            p = os.path.join(raw_dir, sub, f"{image_id}{ext}")
            if os.path.exists(p):
                return p
    return None

def find_mask_file(raw_dir, image_id):
    names = [
        f"{image_id}_segmentation.png",
        f"{image_id}_mask.png",
        f"{image_id}.png",
        f"{image_id}_segmentation.jpg"
    ]
    subdirs = ["", "masks", "ISIC2018_Task1_Training_GroundTruth", "ISIC2018_Task1-2_Training_Input"]
    for sub in subdirs:
        for name in names:
            p = os.path.join(raw_dir, sub, name)
            if os.path.exists(p):
                return p
    return None

def build_ham10000_metadata(raw_dir, out_dir, require_masks=False):
    meta_path = locate_metadata_file(raw_dir)
    if not meta_path:
        raise FileNotFoundError(
            f"Could not locate HAM10000_metadata.csv in {raw_dir}.\n"
            f"Please download the HAM10000 dataset from Harvard Dataverse / ISIC Archive."
        )

    df_raw = pd.read_csv(meta_path)
    if "lesion_id" not in df_raw.columns or "image_id" not in df_raw.columns or "dx" not in df_raw.columns:
        raise ValueError(f"Expected columns lesion_id, image_id, dx in {meta_path}, got: {df_raw.columns.tolist()}")

    out_images_dir = os.path.join(out_dir, "images")
    out_masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)

    records = []
    n_missing_images = 0
    n_missing_masks = 0

    for _, row in df_raw.iterrows():
        img_id = str(row["image_id"]).strip()
        lesion_id = str(row["lesion_id"]).strip()
        dx = str(row["dx"]).strip().lower()

        src_img = find_image_file(raw_dir, img_id)
        if not src_img:
            n_missing_images += 1
            continue

        rel_img_path = f"images/{img_id}.png"
        dst_img_path = os.path.join(out_dir, rel_img_path)
        if not os.path.exists(dst_img_path):
            im = Image.open(src_img).convert("RGB")
            im.save(dst_img_path)

        src_mask = find_mask_file(raw_dir, img_id)
        rel_mask_path = f"masks/{img_id}.png"
        dst_mask_path = os.path.join(out_dir, rel_mask_path)

        if src_mask and os.path.exists(src_mask):
            msk = Image.open(src_mask).convert("L")
            msk.save(dst_mask_path)
            boxes = mask_to_boxes(np.asarray(msk))
        else:
            n_missing_masks += 1
            if require_masks:
                continue
            im = Image.open(dst_img_path)
            msk = Image.new("L", im.size, 0)
            msk.save(dst_mask_path)
            boxes = []

        is_melanoma = 1 if dx == "mel" else 0

        rec = {
            "image_id": img_id,
            "lesion_id": lesion_id,   # authentic HAM10000 lesion identifier
            "patient_id": lesion_id,  # maintained for backward-compatibility with grouped splitters
            "label": is_melanoma,
            "image_path": rel_img_path,
            "mask_path": rel_mask_path,
            "boxes_json": json.dumps(boxes),
            "dx_7class": dx,
            "dx_type": str(row.get("dx_type", "")),
            "age": row.get("age", np.nan),
            "sex": str(row.get("sex", "")),
            "localization": str(row.get("localization", ""))
        }
        records.append(rec)

    df_out = pd.DataFrame(records)
    out_csv = os.path.join(out_dir, "metadata.csv")
    df_out.to_csv(out_csv, index=False)

    summary = {
        "total_records_processed": len(df_out),
        "total_lesions": int(df_out["patient_id"].nunique()) if len(df_out) else 0,
        "melanoma_count": int((df_out["label"] == 1).sum()) if len(df_out) else 0,
        "other_count": int((df_out["label"] == 0).sum()) if len(df_out) else 0,
        "melanoma_prevalence": float(df_out["label"].mean()) if len(df_out) else 0.0,
        "missing_images_skipped": n_missing_images,
        "missing_masks_filled_zeros": n_missing_masks,
        "metadata_csv": out_csv
    }
    return df_out, summary

def main():
    ap = argparse.ArgumentParser(description="HAM10000 + ISIC 2018 Task 1 Data Preprocessor")
    ap.add_argument("--raw_dir", required=True, help="Directory containing HAM10000 images, masks, and metadata")
    ap.add_argument("--out_dir", default="data/ham10000", help="Destination directory for standard metadata.csv")
    ap.add_argument("--require_masks", action="store_true", help="Only keep samples that have ground-truth segmentation masks")
    args = ap.parse_args()

    df_out, summary = build_ham10000_metadata(args.raw_dir, args.out_dir, require_masks=args.require_masks)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
