# __init__.py
from flask import Flask, request, jsonify
from PIL import Image


def create_app(classifier):
    # Generate Flask-Application
    app = Flask(__name__)

    # Define a function that deal with
    # 'POST /  request'.
    @app.route("/", methods=["POST"])
    def predict():
        # Get Received-file handler
        img_file = request.files["img"]

        # Check the file is empty.
        if img_file.filename == "":
            return "Bad Request", 400

        # Read Image-FIle by using 'PIL'.
        img = Image.open(img_file)

        # Predict that is Taco or Burrito
        # by applying Classification-Model.
        result = classifier.predict(img)

        # Return Result as JSON forms.
        return jsonify({
            "result": result
        })

    return app

