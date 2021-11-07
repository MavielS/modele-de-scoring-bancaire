FROM python:3.9

RUN mkdir /backend

ADD . /backend

RUN rm ./backend/rf_objects/X_test_rf
RUN rm ./backend/rf_objects/y_test_rf

WORKDIR /backend

RUN pip install -r requirements.txt
RUN pip install shap

WORKDIR /

# Inter-container interaction
EXPOSE 8080 

# Create a server on 0.0.0.0:8080
CMD ["python", "backend/main.py"]