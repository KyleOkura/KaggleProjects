from models.pytorch_model import NeuralNetwork
from models.xgboost_model import XGBoostModel
from training.train_pytorch import train_pytorch_model
from training.train_xgboost import train_xgboost_model
from utils.device import get_device

def main():
    device = get_device()
    
    # Initialize and train the PyTorch model
    pytorch_model = NeuralNetwork().to(device)
    train_pytorch_model(pytorch_model, device)
    
    # Initialize and train the XGBoost model
    xgboost_model = XGBoostModel()
    train_xgboost_model(xgboost_model)

if __name__ == "__main__":
    main()