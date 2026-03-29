from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build runtime artifacts for the IDS dashboard.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="clean_traffic.csv",
        help="Path to training dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for persisted artifacts.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="Label",
        help="Target label column name.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=120000,
        help="Max number of rows to train on for fast local artifact generation.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=30,
        help="Number of top features to keep after feature importance ranking.",
    )
    return parser.parse_args()


def load_dataset(
    dataset_path: Path,
    sample_size: int,
    label_column: str,
    benign_label: str = "BENIGN",
    chunksize: int = 200000,
) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if sample_size <= 0:
        df = pd.read_csv(dataset_path, low_memory=False)
        df.columns = [str(column).strip() for column in df.columns]
        return df.reset_index(drop=True)

    target_attacks = max(int(sample_size * 0.25), 100)
    target_benign = max(sample_size - target_attacks, 1)

    attack_parts = []
    benign_parts = []
    attack_count = 0
    benign_count = 0
    rng = np.random.default_rng(42)

    for chunk in pd.read_csv(dataset_path, low_memory=False, chunksize=chunksize):
        chunk.columns = [str(column).strip() for column in chunk.columns]

        if label_column not in chunk.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset")

        attack_chunk = chunk[chunk[label_column].astype(str) != benign_label]
        benign_chunk = chunk[chunk[label_column].astype(str) == benign_label]

        if attack_count < target_attacks and not attack_chunk.empty:
            needed_attack = target_attacks - attack_count
            take_attack = min(needed_attack, len(attack_chunk))
            sampled_attack = attack_chunk.sample(
                n=take_attack,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            attack_parts.append(sampled_attack)
            attack_count += len(sampled_attack)

        if benign_count < target_benign and not benign_chunk.empty:
            needed_benign = target_benign - benign_count
            take_benign = min(needed_benign, len(benign_chunk))
            sampled_benign = benign_chunk.sample(
                n=take_benign,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            benign_parts.append(sampled_benign)
            benign_count += len(sampled_benign)

        if attack_count >= target_attacks and benign_count >= target_benign:
            break

    if not attack_parts and not benign_parts:
        raise ValueError("Could not sample rows from dataset.")

    sampled_df = pd.concat([*attack_parts, *benign_parts], ignore_index=True)
    sampled_df = sampled_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return sampled_df


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {dataset_path}")
    df = load_dataset(
        dataset_path=dataset_path,
        sample_size=args.sample_size,
        label_column=args.label_column,
    )

    if args.label_column not in df.columns:
        raise ValueError(f"Label column '{args.label_column}' not found in dataset")

    y_multi = df[args.label_column].astype(str)
    x = df.drop(columns=[args.label_column]).copy()

    x.columns = [str(column).strip() for column in x.columns]
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)

    valid_mask = ~x.isna().any(axis=1)
    x = x.loc[valid_mask]
    y_multi = y_multi.loc[valid_mask]

    y_binary = (y_multi != "BENIGN").astype(int)

    if y_binary.nunique() < 2:
        raise ValueError(
            "Sample does not contain both BENIGN and ATTACK rows. Increase --sample-size or export notebook artifacts."
        )

    if y_multi.nunique() < 2:
        raise ValueError(
            "Sample does not contain multiple attack classes. Increase --sample-size or export notebook artifacts."
        )

    print(f"Training rows after cleaning: {len(x)}")

    x_train, x_test, y_train_multi, y_test_multi, y_train_binary, y_test_binary = train_test_split(
        x,
        y_multi,
        y_binary,
        test_size=0.2,
        random_state=42,
        stratify=y_binary,
    )

    selector = RandomForestClassifier(
        n_estimators=80,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    selector.fit(x_train, y_train_multi)

    importances = pd.Series(selector.feature_importances_, index=x_train.columns)
    selected_features = importances.sort_values(ascending=False).head(args.top_features).index.tolist()

    x_train_sel = x_train[selected_features]
    x_test_sel = x_test[selected_features]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_sel)
    x_test_scaled = scaler.transform(x_test_sel)

    binary_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    binary_model.fit(x_train_scaled, y_train_binary)

    multiclass_model = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
    multiclass_model.fit(x_train_scaled, y_train_multi)

    y_pred_binary = binary_model.predict(x_test_scaled)
    y_pred_multi = multiclass_model.predict(x_test_scaled)

    binary_accuracy = float(accuracy_score(y_test_binary, y_pred_binary))
    multi_accuracy = float(accuracy_score(y_test_multi, y_pred_multi))

    print(f"Binary accuracy : {binary_accuracy:.4f}")
    print(f"Multi accuracy  : {multi_accuracy:.4f}")

    joblib.dump(binary_model, output_dir / "binary_model.pkl")
    joblib.dump(multiclass_model, output_dir / "multiclass_model.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")
    joblib.dump(selected_features, output_dir / "feature_list.pkl")

    metadata = {
        "dataset": str(dataset_path.name),
        "rows_used": int(len(x)),
        "selected_feature_count": len(selected_features),
        "binary_accuracy": binary_accuracy,
        "multi_accuracy": multi_accuracy,
        "binary_classes": [str(value) for value in sorted(y_binary.unique().tolist())],
        "multi_classes": sorted(y_multi.unique().tolist()),
        "note": "Artifacts generated by scripts/build_artifacts.py. Replace with notebook artifacts for final parity.",
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)

    print("Saved artifacts:")
    print(f"  - {output_dir / 'binary_model.pkl'}")
    print(f"  - {output_dir / 'multiclass_model.pkl'}")
    print(f"  - {output_dir / 'scaler.pkl'}")
    print(f"  - {output_dir / 'feature_list.pkl'}")
    print(f"  - {output_dir / 'metadata.json'}")

    print("\nBinary report:")
    print(classification_report(y_test_binary, y_pred_binary, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
