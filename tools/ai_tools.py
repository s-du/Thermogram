from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from autodistill.utils import plot

def run_g_dino(image_path, ontology):
    base_model = GroundingDINO(ontology=CaptionOntology(ontology))

    results = base_model.predict(image_path)

    return results