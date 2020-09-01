FROM lucidfrontier45/pytorch

RUN mkdir /FlaskAPI
WORKDIR /FlaskAPI

COPY requirements.txt /FlaskAPI
RUN pip install --no-cache-dir -r requirements.txt

COPY app /FlaskAPI/app/app
COPY tack_burrito.prm /FlaskAPI/
COPY wsgi.py /FlaskAPI/

CMD gunicorn --access-logfile - \
            -b 0.0.0.0:8000 -w 4 \
            --preload wsgi:app


