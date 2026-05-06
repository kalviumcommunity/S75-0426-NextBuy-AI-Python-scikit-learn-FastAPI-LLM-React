from src.train import train
from src.evaluate import evaluate

def main():
    model, pipeline, X_test, y_test = train()
    evaluate(model, pipeline, X_test, y_test)

if __name__ == "__main__":
    main()