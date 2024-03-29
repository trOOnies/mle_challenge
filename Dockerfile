# syntax=docker/dockerfile:1.2
# I didn't use a Dockerfile at the end, but here's how I'd do it
# (Take into account that I should change the code to reference a 'mle_challenge' folder for it to work)
FROM python:3.8.19-slim
WORKDIR /
RUN python -m pip install --upgrade pip
ADD ./requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN mkdir /mle_challenge
ADD ./challenge /mle_challenge/ && ./data /mle_challenge/
EXPOSE 8080
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
