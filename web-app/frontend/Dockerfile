FROM python:3.9

RUN mkdir /frontend

ADD . /frontend

WORKDIR /frontend

RUN pip install -r requirements.txt
RUN pip install shap


# Inter-container interaction
EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py"]