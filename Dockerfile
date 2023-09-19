FROM continuumio/anaconda3:4.4.0

RUN pip install -r requirements.txt

EXPOSE 5000

CMD streamlit run app.py 
