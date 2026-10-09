# IMDb Sentiment Classification with a GPT-style Transformer

A transformer built in PyTorch and trained from scratch to classify IMDb
movie reviews as positive or negative.

## Dataset
- IMDb Large Movie Review Dataset: 25,000 train / 25,000 test reviews
- Train set split into 22,500 train / 2,500 validation
- Balanced classes (50% positive, 50% negative)
- Reviews tokenized and truncated/padded to 128 tokens

## Model
`DemoGPT`: token and positional embeddings, stacked transformer blocks with
self-attention, layer norm, mean pooling, and a linear classifier (2 classes).

## Training
- Loss: CrossEntropyLoss
- Optimizer: AdamW (learning rate 3e-4)
- 3 epochs, batch size 32

## Results

| Epoch | Validation Accuracy |
|---|---|
| 1 | 71.80% |
| 2 | 75.76% |
| 3 | 78.08% |

Test accuracy: **XX.XX%** (chance level is 50%)

## How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Download the IMDb dataset and place it in `aclImdb/`
4. Open the notebook and run all cells

## Next Steps
- Longer sequence length, larger model, learning-rate schedule
- Fine-tune a pretrained model such as DistilBERT

## Author
Yvonne Gitau