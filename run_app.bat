@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Make sure your model files are in: ./bert_sentiment/final/
echo.
echo Starting Streamlit app...
streamlit run app.py
