web: gunicorn -w 2 -k uvicorn.workers.UvicornWorker server:app --timeout 120 --bind 0.0.0.0:$PORT
