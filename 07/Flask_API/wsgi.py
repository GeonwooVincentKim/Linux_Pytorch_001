# wsgi.py
import torch
from smart_getenv import getenv

from app import create_app
from app.classifier import Classifier

# Read Parameter File-directory in the
# Environment-Variable.
prm_file = getenv("PRM_FILE", default="./taco_burrito.prm")
print(prm_file)

# Read Parameter-File.
params = torch.load(
    prm_file,
    map_location=lambda storage,
    loc: storage
)


if __name__ == "__main__":
    # Generate Classifier and Flask Application.
    classifier = Classifier(params)
    app = create_app(classifier)

