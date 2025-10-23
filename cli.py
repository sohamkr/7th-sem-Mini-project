
import argparse
from run_all import run_all
from config import OUT_DIR
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--mode', type=str, default='all', choices=['all','seq','xgb','arima'], help='Which pipeline to run')
    args = parser.parse_args()

    if args.mode == 'all':
        run_all(args.data)
    else:
       
        print("Running full pipeline (mode flags not yet separated).")
        run_all(args.data)

if __name__ == "__main__":
    main()
