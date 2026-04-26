FROM python:3.10-slim
RUN pip install pytest pytest-flakefinder
WORKDIR /workspace
