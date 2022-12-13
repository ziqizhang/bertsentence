from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging, sys
logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger("semsim")

class Semsim:
    def __init__(self, model_path_or_name):
        log.info("Loading model...")
        self.model=SentenceTransformer(model_path_or_name)
    def encode(self, sentences:list):
        log.info("Encoding sentences...")
        return self.model.encode(sentences)

    def similarity(selfs, list1, list2):
        return cosine_similarity(list1, list2)
