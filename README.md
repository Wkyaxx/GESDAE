# GESDAE

Graph Embedded Semi-supervised Deep Autoencoder for Fault Diagnosis in Rotating Machinery

## Overview
GESDAE (Graph Embedded Semi-supervised Deep Autoencoder) is a novel deep learning model designed for **fault diagnosis** in rotating machinery, specifically focusing on handling high-dimensional, sparse fault data. The model combines **graph embedding techniques** with **semi-supervised deep autoencoders** to effectively learn both local and global feature relationships, making it suitable for both single and compound faults. This repository contains the code for training and evaluating GESDAE on fault diagnosis tasks.

For verification, you can access the GitHub repository here: [GESDAE on GitHub](https://github.com/Wkyaxx/GESDAE)

## Features
- **Graph Embedding**: Preserves both local and global relationships in data, improving the ability to handle complex fault patterns.
- **Semi-supervised Learning**: Integrates unsupervised and supervised learning to refine feature extraction based on class labels.
- **Sparse Dropout Regularization**: Enhances the model's robustness by removing less important features during training, preventing overfitting.
- **Laplacian Regularization**: Regularizes the graph structure to promote better overall feature representation.

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- PyWavelets (for wavelet transform)

You can install the required dependencies using pip:
```bash
pip install -r requirements.txt

## Directory Structure
/GESDAE
    ├── data/                  # Directory for data files
    ├── models/                # Model definition files
    ├── scripts/               # Training and testing scripts
    ├── utils/                 # Utility functions (e.g., data preprocessing)
    ├── requirements.txt       # Required Python libraries
    └── README.md              # Project documentation

## Data Format
GESDAE is designed to work with data formatted as follows:

Input Data: The input data is assumed to be a .xlsx file with each row representing a sample, and columns representing features. The last column is considered as the label (fault type).
Feature Columns: Time-domain, frequency-domain, and time-frequency domain features (such as mean, variance, RMS, FFT, etc.) are used as inputs to the model.
Label Column: The last column should contain fault class labels.

## Example Input:
Feature1	Feature2	...	FeatureN	Label
0.234	0.456	...	0.789	F1
0.123	0.678	...	0.901	F5
...	...	...	...	...

## Training the Model
Prepare the Data:
Place your dataset (data.xlsx) into the data/ folder.
Make sure that the data follows the format mentioned above (features and labels).
Run the Training Script: Use the following command to start training the model:
python scripts/train_model.py --data_path data/data.xlsx --epochs 50 --batch_size 128 --learning_rate 0.01 --embed_dim 5
--data_path: Path to the data file.
--epochs: Number of training epochs (default is 50).
--batch_size: Batch size for training (default is 128).
--learning_rate: Learning rate for Adam optimizer (default is 0.01).
--embed_dim: Dimensionality of the embedding space (default is 5).
Evaluate the Model: After training, the model can be evaluated on the test set. The evaluation process is done automatically after training and results in classification accuracy.
python scripts/evaluate_model.py --data_path data/data.xlsx --model_path models/trained_model.pth
--data_path: Path to the test data.
--model_path: Path to the saved trained model.

## Hyperparameter Tuning
Grid Search: We used grid search to tune key hyperparameters such as batch size, learning rate, and sparsity parameter (r).
Optimal Settings:
Batch size: 128
Learning rate: 0.01
Sparsity parameter (r): 0.3
To adjust these values, you can modify the script or use hyperparameter optimization techniques to explore additional configurations.

## Model Architecture
GESDAE consists of three main components:
Graph Embedding: This part learns the relationships between fault features using graph-based methods, which preserve both local and global data structures.
Deep Autoencoder: The autoencoder is used for feature extraction and dimensionality reduction. It consists of three fully connected layers:
First layer: Linear transformation to a 256-dimensional space.
Second layer: Linear transformation to a 128-dimensional space.
Final layer: Embedding layer of size embed_dim (set to 5 in our experiments).
Regularization:
Sparse Dropout: A form of regularization that randomly removes features with lower activation values.
Laplacian Regularization: Used to regularize the graph structure, ensuring that the model does not overfit.
Example Result:
Epoch [50/50], Loss: 0.1523
Optimized KNN Classification Accuracy: 97.30%

## Strategies for Improvement
Data Preprocessing: For better performance, optimize data preprocessing steps, especially for faults with subtle or simple features.
Ensemble Learning: Use ensemble techniques like stacking and boosting to further improve classification accuracy.
Model Architecture Enhancements: Integrate convolutional layers, attention mechanisms, or RNNs to improve the model’s ability to capture hierarchical or sequential relationships in the fault data.

## Contribution
Feel free to fork this repository, make improvements, and submit pull requests. We encourage contributions that enhance model performance, scalability, and usability.
