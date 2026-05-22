# Training Setup

The training process was executed in two phases: primary training and continuation training.

## 1. Hyperparameters
* **Batch Size**: 64
* **Dropout**: 0.3
* **Gradient Clipping**: $\max ||g||_2 = 1.0$ (to stabilize LSTM gradients)
* **Teacher Forcing Ratio**: 0.5 (scheduled sampling helper)

## 2. Optimization Schedule
* **Primary Training (Epochs 1-15)**: 
  * Optimizer: Adam
  * Learning Rate: $3 	imes 10^{-4}$
* **Continuation Training (Epoch 16)**:
  * Optimizer: Adam
  * Learning Rate: $1.5 	imes 10^{-4}$ (reduced by 2x for convergence stability)
  * Resumed from the Best Validation loss checkpoint (Epoch 13, Loss: 6.493)

## 3. Key Observations
* Validation loss plateaued around epoch 13.
* During continuation training, validation loss remained stable, but generation-level metrics (BLEU and CHRF) showed consistent progression, indicating that the model was refining sequence-level decoding rules.
