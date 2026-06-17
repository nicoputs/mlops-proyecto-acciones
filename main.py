import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "predict"):
        print("Uso: python main.py [train|predict]")
        sys.exit(1)
    if sys.argv[1] == "train":
        import train

        train.main()
    else:
        import predict

        predict.main()


if __name__ == "__main__":
    main()
