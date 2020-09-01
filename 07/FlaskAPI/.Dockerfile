FROM lucidfrontier45/pytorch

RUN mkdir /FlaskAPI
WORKDIR /FlaskAPI

COPY requirements.txt /FlaskAPI
