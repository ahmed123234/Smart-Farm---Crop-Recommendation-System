FROM python:3.11-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile",  "Pipfile.lock",  "./"]

RUN pipenv install --system --deploy 

COPY ["predict.py", "crop_recommendation_model_pipeline.pkl", "./"]

EXPOSE 9040

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9040", "predict:app"]
