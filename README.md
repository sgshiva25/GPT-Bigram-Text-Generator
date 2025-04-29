# GPT Language Model Based on Bigram Model

This project implements a GPT-like language model using PyTorch, designed for training on text data and generating new text based on learned patterns. The model is based on a bigram model, where the prediction of the next word is conditioned on the previous word, and this dependency is captured through a transformer-based architecture.

## Features

- **Transformer Architecture**: The model is based on the Transformer architecture, leveraging self-attention mechanisms.
- **Multi-Head Attention**: The self-attention mechanism is implemented with multiple attention heads to capture different relationships in the data.
- **Bigram-Based Model**: The model operates on a bigram approach, learning the relationship between consecutive words for text generation.
- **Text Generation**: The model is capable of generating new text based on a given input sequence.
- **Training and Evaluation**: The model supports training on both training and validation datasets and evaluates the loss during training.


## Dataset

This model requires a text dataset with a `vocab.txt` file containing the vocabulary and training/validation data files like `train_split.txt` and `val_split.txt`.

## Usage

### 1. Training the Model

Run the script to train the GPT model on your dataset:

```bash
python train.py
```

You can adjust hyperparameters like `batch_size`, `block_size`, and `learning_rate` directly in the script.

### 2. Saving the Model

After training, the model’s weights will be saved in the `model-01.pth` file. This file contains the state dictionary of the trained model.

### 3. Loading the Model

You can later load the saved model for evaluation or further training:

```python
model.load_state_dict(torch.load('model-01.pth', map_location=torch.device('cpu')))
model.eval()
```

### 4. Text Generation

Use the `generate()` method to generate new text sequences based on a given input:

```python
generated_text = model.generate(input_sequence, max_new_tokens=100)
```

## Code Structure

- **Head**: Implements one head of the self-attention mechanism.
- **MultiHeadAttention**: Implements multiple heads of self-attention in parallel.
- **FeedForward**: Implements the feed-forward network after the attention mechanism.
- **Block**: Implements a Transformer block, which includes multi-head attention and feed-forward layers.
- **GPTLanguageModel**: Implements the complete GPT-like language model, including token embeddings, positional embeddings, and Transformer blocks. The model is based on a bigram model, predicting the next token based on the previous one.

## Training Loop

The training loop involves:
1. Sampling a batch of training data.
2. Computing the forward pass and loss.
3. Performing backpropagation to update model parameters.
4. Periodically evaluating the model's performance on the validation set.

## Evaluation

The model's loss is evaluated periodically using the `estimate_loss()` function, which computes the mean loss for both the training and validation datasets.

