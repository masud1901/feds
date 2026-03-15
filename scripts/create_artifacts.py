"""Create initial global model (.npz) and K files (JSON). Run from repo root.
Usage: python scripts/create_artifacts.py [--dataset mnist|cifar10|speech|all]
No pickle: models are NumPy .npz, K values are JSON."""
import os
import sys
import argparse
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
SCRIPTS = os.path.join(REPO, "scripts")


def run(script_name):
    path = os.path.join(SCRIPTS, script_name)
    if os.path.exists(path):
        subprocess.run([sys.executable, path], cwd=REPO, check=True)
    else:
        print(f"Skip {script_name} (not found)")


def main():
    parser = argparse.ArgumentParser(description="Create initial global model and K pickle")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "speech", "all"], default="all")
    args = parser.parse_args()
    if args.dataset in ("mnist", "all"):
        run("create_artifacts_colab.py")
    if args.dataset in ("cifar10", "all"):
        run("create_artifacts_cifar.py")
    if args.dataset in ("speech", "all"):
        run("create_artifacts_Speech.py")


if __name__ == "__main__":
    main()
