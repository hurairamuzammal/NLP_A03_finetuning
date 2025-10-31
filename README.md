# Sentiment Analysis Streamlit App

## Setup Instructions

### 1. Download Model from Kaggle

Since you trained your model on Kaggle with checkpoints, you need to download the model files:

**Option A: Download from Kaggle Output**
1. Go to your Kaggle notebook
2. Click on the "Output" tab in the right sidebar
3. Find the `bert_sentiment/final` folder
4. Download the entire folder
5. Extract it to this project directory so you have: `./bert_sentiment/final/`

**Option B: Use Kaggle API**
```bash
pip install kaggle
kaggle kernels output <your-username>/<notebook-name> -p ./
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

## Model Files Required

The `./bert_sentiment/final/` directory should contain:
- `config.json`
- `model.safetensors` (or `pytorch_model.bin`)
- `tokenizer_config.json`
- `vocab.txt`
- `special_tokens_map.json`

## Usage

### Single Text Analysis
- Enter any text in the text area
- Click "Analyze Sentiment"
- View the predicted sentiment and confidence score

### Batch Analysis
- Enter multiple texts (one per line)
- Click "Analyze Batch"
- View results in a table
- Download results as CSV

## Features

- Real-time sentiment analysis
- Confidence scores
- Probability distribution visualization
- Batch processing
- CSV export
- GPU support (if available)
