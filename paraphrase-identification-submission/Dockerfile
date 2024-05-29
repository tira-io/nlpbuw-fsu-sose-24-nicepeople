FROM fschlatt/natural-language-processing-exercises:0.0.1

# Install specific version of scikit-learn
RUN pip install scikit-learn==1.4.2

ADD run.py /code/run.py
ADD model.joblib /code/model.joblib
ADD vectorizer.joblib /code/vectorizer.joblib

ENTRYPOINT [ "python3", "/code/run.py" ]