import argparse
from _bert_model import Model as Bert
from _t5_model import Model as T5


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train, evaluate or optimize a language model.")
    
    parser.add_argument(
        "-model",
        dest="model",
        help="Select the language model (e.g., 'bert-base-uncased', 't5-large')",
        required=True
    )
    parser.add_argument(
        "-train",
        dest="train",
        help="Enable model training",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "-evaluate",
        dest="evaluate",
        help="Enable model evaluation on a test set",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "-optuna",
        dest="optuna",
        help="Enable Optuna hyperparameter optimization",
        action="store_true",
        default=False
    )
    
    return parser.parse_args()


def get_model(model_name: str):
    """Instantiate and return the appropriate model based on the model name."""
    if 'bert' in model_name.lower():
        return Bert(model_name)
    elif 't5' in model_name.lower():
        return T5(model_name)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")


def main():
    """Main function to handle model training, evaluation, and optimization."""
    args = parse_arguments()
    
    model = get_model(args.model)
    if args.train:
        print("Training the model...")
        model.train()
    if args.evaluate:
        print("Evaluating the model...")
        model.predict()
    if args.optuna:
        print("Running Optuna optimization...")
        model.optuna()


if __name__ == "__main__":
    main()
