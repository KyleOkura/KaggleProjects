# Neural XGBoost Project

This project combines a PyTorch neural network model with an XGBoost model for enhanced predictive performance. The goal is to leverage the strengths of both machine learning frameworks to achieve better results on various datasets.

## Project Structure

```
neural-xgboost-project
├── src
│   ├── app.py                # Main entry point for the application
│   ├── models
│   │   ├── pytorch_model.py   # Defines the PyTorch neural network model
│   │   └── xgboost_model.py    # Defines the XGBoost model
│   ├── training
│   │   ├── train_pytorch.py    # Training script for the PyTorch model
│   │   └── train_xgboost.py     # Training script for the XGBoost model
│   ├── utils
│   │   └── device.py           # Utility functions for device management
│   └── types
│       └── __init__.py         # Custom types and interfaces
├── requirements.txt            # Project dependencies
├── pyproject.toml              # Project configuration
└── README.md                   # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd neural-xgboost-project
   ```

2. **Install dependencies:**
   Use pip to install the required packages listed in `requirements.txt`.
   ```
   pip install -r requirements.txt
   ```

3. **Check GPU availability:**
   Ensure that your environment is set up to utilize the NVIDIA 4060 GPU. The project includes utility functions to check and set the device for PyTorch.

## Usage

- To train the PyTorch model, run:
  ```
  python src/training/train_pytorch.py
  ```

- To train the XGBoost model, run:
  ```
  python src/training/train_xgboost.py
  ```

## Models

- **Neural Network (PyTorch):** The `NeuralNetwork` class in `src/models/pytorch_model.py` defines the architecture and training methods for the neural network.

- **XGBoost Model:** The `XGBoostModel` class in `src/models/xgboost_model.py` encapsulates the training and evaluation processes for the XGBoost model.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.