"""
CLI entry-point: extract grayscale line-art sketches from raw anime images.

Options
-------
    --input-dir DIR        Source directory containing raw anime images.
                           (default: $INPUT_DIR or data/raw/portraits)
    --output-dir DIR       Destination for extracted sketches.
                           (default: $OUTPUT_DIR or data/processed/sketches)
    --extractor NAME       Sketch extractor: lineart-anime or anime2sketch.
    --detect-resolution N  ControlNet detector input resolution (default: 512).
    --image-resolution N   Output/load resolution (default: 512).
    --max-images N         Max new images to process in this run (default: 100).
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from utils.paths import DEFAULT_RAW_IMAGE_DIR, DEFAULT_SKETCHES_DIR
from pipeline.lineart import (
    ANIME2SKETCH,
    LINEART_ANIME,
    SUPPORTED_EXTRACTORS,
    check_preprocessing_config,
    collect_images,
    create_extractor,
    extractor_output_name,
    process_image,
    write_preprocessing_config,
)

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract lineart sketches from anime images."
    )
    parser.add_argument(
        "--input-dir",
        default=os.getenv(
            "INPUT_DIR",
            str(DEFAULT_RAW_IMAGE_DIR),
        ),
        help="Directory containing raw anime images.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv(
            "OUTPUT_DIR",
            str(DEFAULT_SKETCHES_DIR),
        ),
        help="Directory where extracted sketches will be saved.",
    )
    parser.add_argument(
        "--extractor",
        choices=SUPPORTED_EXTRACTORS,
        default=os.getenv("SKETCH_EXTRACTOR", LINEART_ANIME),
        help=(
            "Sketch extractor to use. lineart-anime keeps the current "
            "ControlNet baseline; anime2sketch uses a local Anime2Sketch "
            "checkout (default: lineart-anime)."
        ),
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        default=os.getenv("FLAT_OUTPUT", "0") == "1",
        help=(
            "Write directly to output-dir. By default, sketches are written "
            "under output-dir/<extractor_name>/ so extractors can be compared."
        ),
    )
    parser.add_argument(
        "--detect-resolution",
        type=int,
        default=int(os.getenv("DETECT_RES", "512")),
        help="Resolution passed to ControlNet detectors (default: 512).",
    )
    parser.add_argument(
        "--image-resolution",
        type=int,
        default=int(os.getenv("IMAGE_RES", "512")),
        help="Output resolution for ControlNet, load size for Anime2Sketch (default: 512).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help=(
            "Maximum number of new images to process in this run. "
            "Use 0 to process all pending images (default: 100)."
        ),
    )
    parser.add_argument(
        "--max-per-folder",
        dest="max_per_folder",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--anime2sketch-dir",
        default=os.getenv("ANIME2SKETCH_DIR"),
        help=(
            "Path to a local Mukosame/Anime2Sketch checkout. Required when "
            "--extractor anime2sketch is selected."
        ),
    )
    parser.add_argument(
        "--anime2sketch-python",
        default=os.getenv("ANIME2SKETCH_PYTHON"),
        help=(
            "Python executable for Anime2Sketch. Defaults to the current "
            "Python interpreter."
        ),
    )
    parser.add_argument(
        "--anime2sketch-model",
        choices=("default", "improved"),
        default=os.getenv("ANIME2SKETCH_MODEL", "default"),
        help="Anime2Sketch model variant (default: default).",
    )
    parser.add_argument(
        "--anime2sketch-gpu-ids",
        default=os.getenv("ANIME2SKETCH_GPU_IDS", ""),
        help="Optional Anime2Sketch GPU ids, for example 0 or 0,1.",
    )
    parser.add_argument(
        "--anime2sketch-clahe-clip",
        type=float,
        default=float(os.getenv("ANIME2SKETCH_CLAHE_CLIP", "-1")),
        help="Anime2Sketch CLAHE clip limit; values <= 0 disable it.",
    )

    parser.add_argument(
        "--no-autocrop",
        action="store_true",
        default=False,
        help=(
            "Disable automatic cropping of black letterbox/pillarbox borders. "
            "Use this when resuming extraction on a directory that was started "
            "without autocrop, or when inputs have dark backgrounds that should "
            "not be cropped."
        ),
    )
    args = parser.parse_args()
    if args.max_images is not None and args.max_per_folder is not None:
        parser.error("--max-images and --max-per-folder cannot be used together.")

    args.used_deprecated_max_per_folder = args.max_per_folder is not None
    if args.max_images is None:
        default_max = os.getenv("MAX_IMAGES", os.getenv("MAX_PER_FOLDER", "100"))
        args.max_images = args.max_per_folder
        if args.max_images is None:
            args.max_images = int(default_max)

    if args.max_images < 0:
        parser.error("--max-images must be 0 or greater.")

    return args


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_dir).resolve()
    output_base = Path(args.output_dir).resolve()
    output_root = (
        output_base
        if args.flat_output
        else output_base / extractor_output_name(args.extractor)
    )

    if not input_root.exists():
        print(f"[ERROR] Input directory not found: {input_root}", file=sys.stderr)
        sys.exit(1)

    if args.extractor == ANIME2SKETCH and not args.anime2sketch_dir:
        print(
            "[ERROR] --extractor anime2sketch requires --anime2sketch-dir "
            "or ANIME2SKETCH_DIR.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        extractor = create_extractor(
            args.extractor,
            detect_resolution=args.detect_resolution,
            image_resolution=args.image_resolution,
            autocrop=not args.no_autocrop,
            anime2sketch_dir=args.anime2sketch_dir,
            anime2sketch_python=args.anime2sketch_python,
            anime2sketch_model=args.anime2sketch_model,
            anime2sketch_gpu_ids=args.anime2sketch_gpu_ids,
            anime2sketch_clahe_clip=args.anime2sketch_clahe_clip,
        )
    except Exception as exc:
        print(f"[ERROR] Could not initialize extractor: {exc}", file=sys.stderr)
        sys.exit(1)

    # Autocrop is only implemented by ControlNet.  For other extractors
    # the effective setting is always False regardless of the CLI flag,
    # so the preprocessing marker accurately reflects the actual behavior.
    autocrop_requested = not args.no_autocrop
    autocrop_enabled = autocrop_requested and args.extractor == LINEART_ANIME
    if autocrop_requested and not autocrop_enabled:
        print(
            f"[info] --no-autocrop not needed: autocrop is only supported "
            f"by the {LINEART_ANIME} extractor."
        )
    check_preprocessing_config(output_root, autocrop=autocrop_enabled)
    write_preprocessing_config(output_root, autocrop=autocrop_enabled)

    print(f"[init] Scanning {input_root} …")
    print(f"[init] Extractor: {args.extractor}")
    print(f"[init] Autocrop: {'enabled' if autocrop_enabled else 'disabled'}")
    print(f"[init] Output: {output_root}")
    if args.used_deprecated_max_per_folder:
        print("[warn] --max-per-folder is deprecated; use --max-images instead.")
    limit_text = (
        "all pending images" if args.max_images == 0 else f"{args.max_images} new images"
    )
    print(f"[init] Limit: {limit_text}.")
    pairs = collect_images(input_root, output_root, max_images=args.max_images)
    total = len(pairs)
    print(f"[init] {total} images to process (already-done files are skipped).")

    if total == 0:
        print("[done] Nothing to do.")
        return

    ok = skipped = 0
    with tqdm(pairs, unit="img", dynamic_ncols=True) as bar:
        for src, dst in bar:
            bar.set_postfix_str(src.name)
            success = process_image(src, dst, extractor)
            if success:
                ok += 1
            else:
                skipped += 1

    print(f"\n[done] Processed {ok}/{total} images ({skipped} skipped).")
    print(f"[done] Sketches saved to: {output_root}")


if __name__ == "__main__":
    main()
