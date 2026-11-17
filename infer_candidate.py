"""TRACE: reproduce frame-level possession labels from automatic tracking."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from candidate_pipeline import digest, infer, load_features, load_models, validate_package, write_json_exclusive
from documented_structure import apply_documented_dead
from games import GAME


def run(data_dir, feature_file, output):
    started = time.perf_counter()
    _, runtime = validate_package()
    output = Path(output).resolve()
    receipt = output.with_suffix(output.suffix + ".verification.json")
    if output.exists() or receipt.exists():
        raise FileExistsError("Output/receipt already exists")
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("Write generated files outside the package or beneath outputs/")
    if feature_file is not None:
        inputs = {"predictor_archive": Path(feature_file).resolve()}
        before = {name: digest(path) for name, path in inputs.items()}
        features, blocks = load_features(feature_file)
        input_level = "precomputed predictor features"
    else:
        # Uses unchanged public organizer frame/period definitions. No labels,
        # videos, corrected tracking, Smart Tagging or homography are requested.
        from games import GAME, GAMES, half_video_blocks, require, source_frames
        from research_possession.tracking import read_atd
        metadata, raw = map(Path, require(["atd_meta", "atd_raw"], data_dir=str(Path(data_dir).resolve())))
        inputs = {"automatic_tracking_metadata": metadata, "automatic_tracking_raw": raw}
        before = {name: digest(path) for name, path in inputs.items()}
        features = read_atd(metadata, raw, source_frames(), GAMES[GAME]["fps"], GAMES[GAME]["pitch"])
        blocks = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
        input_level = "raw automatic tracking"
    predictions, audit = infer(features, blocks, load_models())
    if {name: digest(path) for name, path in inputs.items()} != before:
        raise ValueError("Input files changed during inference")
    labels, audit["documented_structure"] = apply_documented_dead(predictions["constrained"], blocks, GAME)
    validate_package()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        stream.write("frame,label\n")
        stream.writelines(f"{i},{label}\n" for i, label in enumerate(labels))
    result = {"status": "TRACE PREDICTIONS GENERATED", "input_level": input_level,
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "rows": len(features), "csv_sha256": digest(output), "runtime": runtime,
              "input_sha256": before, "elapsed_seconds": time.perf_counter() - started,
              "artifact_manifest_sha256": digest(ROOT / "artifacts.json"), **audit}
    write_json_exclusive(receipt, result)
    print("Wrote TRACE possession CSV and verification receipt.", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dir", type=Path, help="Authorized organizer-style automatic-tracking bundle")
    source.add_argument("--features", type=Path, help="Documented predictor-only feature archive")
    parser.add_argument("--output", type=Path, required=True, help="New local possession CSV")
    args = parser.parse_args()
    run(args.data_dir, args.features, args.output)


if __name__ == "__main__":
    main()
