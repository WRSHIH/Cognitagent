FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./app .
RUN chown -R appuser:appgroup /app
USER appuser
ENV PORT 8080
EXPOSE 8080
CMD ["python", "ui.py"]