# Customer Order Prediction LSTM Model

## Project Overview
This project was a proof of concept to show open source deep learning packages can be used to predict customer orders. A Long Short-Term Memory (LSTM) neural network was trained on customer historical order sequences to predict the probability a customer would place an order in the next 7 days. 

## Features
- Sequence-based prediction using LSTM architecture
- Custom data generators for efficient batch processing
- User-based train/validation split to test generalization across customers
- Handling of class imbalance through weighted loss function
- Early stopping and learning rate scheduling for optimal training

## Key Components

### EDA
- Initial EDA was completed in the MSDS6825_practicum_EDA.ipynb file to better understand historical order data and customer data.
-  
### Data Preprocessing
- User sampling to reduce computational load (50% of users with 5+ orders)
- Feature engineering including:
  - Days since last order
  - Customer age at purchase time
  - Account age at purchase time
  - Order pricing features
  - Average days between order for each sequence built into the generator

### Model Architecture
- Multi-layer LSTM network with:
  - 3 LSTM layers (64, 32, and 16 units)
  - Batch normalization between layers
  - L2 regularization to prevent overfitting
  - Earlier iterations used dropout, but better validation performance was acheived without them
  - Final sigmoid activation for binary prediction

### Training Strategy
- Customer-based train/validation split (80/20)
- Variable sequence length support (minimum 5 orders, target 10 orders)
- Binary cross-entropy loss with class weighting
- Progressive learning rate reduction
- Early stopping to prevent overfitting

## Performance
The model achieves:
- Validation accuracy: ~80%
- Validation AUC: ~0.85
- Validation precision: ~0.65
- Validation recall: ~0.72

## Requirements
- TensorFlow 2.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Customization
- Adjust `sequence_length` to change the number of orders used in prediction
- Modify `batch_size` and `epochs` for training optimization
- Tune the model architecture in `create_lstm_model()` for different use cases

## Future Improvements
- Implementation of more features (historical text messages, order wait time adjusted for schedule deliveries)
- Regression approach to predict exact days until next order
- Full dataset training for production use
