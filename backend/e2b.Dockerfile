FROM e2bdev/code-interpreter:latest 

RUN pip install pandas numpy matplotlib seaborn nltk scikit-learn spacy textblob gensim wordcloud statsmodels plotly openpyxl xlrd pyarrow fastparquet requests tqdm joblib 