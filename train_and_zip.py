import os
import shutil
import argparse
import subprocess
import sys


def train_and_zip(args):
    train_cmd = [
        sys.executable, "train_new_topic_generator.py",
        "--data_path", args.data_path,
        "--title_col", args.title_col,
        "--domain_col", args.domain_col,
        "--summary_col", args.summary_col,
        "--model_name", args.model_name,
        "--output_dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--max_input_length", str(args.max_input_length),
        "--max_target_length", str(args.max_target_length),
    ]

    if args.subset_size:
        train_cmd += ["--subset_size", str(args.subset_size)]

    if args.target_col:
        train_cmd += ["--target_col", args.target_col]

    print("\n=== STARTING TRAINING ===\n")
    result = subprocess.run(train_cmd)

    if result.returncode != 0:
        print("\nTraining failed. Skipping zip.")
        sys.exit(1)

    print("\n=== TRAINING COMPLETE. ZIPPING MODEL... ===\n")

    zip_path = shutil.make_archive(
        base_name=args.output_dir,
        format="zip",
        root_dir=".",
        base_dir=args.output_dir
    )

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Model zipped successfully: {zip_path}  ({size_mb:.1f} MB)")
    print("\nTo use for prediction, unzip and point MODEL_PATH to the folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--title_col", type=str, default="title")
    parser.add_argument("--domain_col", type=str, default="category")
    parser.add_argument("--summary_col", type=str, default="summary")
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base")
    parser.add_argument("--output_dir", type=str, default="flant5_topic_model")
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=64)

    args = parser.parse_args()
    train_and_zip(args)
