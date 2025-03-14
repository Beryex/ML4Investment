import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", type=str, default="AAPL")
    args = parser.parse_args()
    print(args.stock)