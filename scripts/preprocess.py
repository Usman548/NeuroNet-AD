import argparse

from neuronet_ad.data_preprocessing import preprocess_oasis


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess OASIS .nii files to multi-view PNG slices")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to OASIS source root")
    parser.add_argument("--output-dir", type=str, default="Output", help="Path to save PNG slices")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["Converted", "Demented", "Nondemented"],
        help="Class folders to process",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    preprocess_oasis(args.input_dir, args.output_dir, args.classes)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()

