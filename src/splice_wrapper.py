"""SpLiCE with dense per-concept cosine-similarity embeddings.

This is the "nonsparse" embedding used throughout the Merlin-Arthur SpLiCE
experiments: for each image, the ``(num_concepts,)`` vector of cosine
similarities between the (centered, normalized) CLIP image embedding and every
concept in the SpLiCE dictionary -- instead of SpLiCE's sparse decomposition
weights.

Depends on upstream SpLiCE (github.com/AI4LIFE-GROUP/SpLiCE, installed via pip).
The ONLY change vs. upstream is the ``encode_image`` override below; the concept
dictionary, vocabulary and image means are loaded/downloaded by upstream as-is.
"""
import torch
import splice
from splice import *  # re-export upstream API (load, get_vocabulary, get_preprocess, ...)
from splice.model import SPLICE


class SPLICEDense(SPLICE):
    """SPLICE variant whose ``encode_image`` returns dense per-concept similarities."""

    def encode_image(self, image):
        if self.clip is not None:
            self.clip.eval()
            with torch.no_grad():
                image = self.clip.encode_image(image)
        image = torch.nn.functional.normalize(image, dim=1)
        centered_image = torch.nn.functional.normalize(image - self.image_mean, dim=1)
        similarity_scores = centered_image @ self.dictionary.T
        if self.return_weights:
            return self.decompose(centered_image), similarity_scores
        return similarity_scores


def load(*args, **kwargs):
    """Drop-in for ``splice.load`` that returns dense per-concept similarities.

    Shadows the upstream ``load`` pulled in by ``from splice import *`` above.
    """
    model = splice.load(*args, **kwargs)
    model.__class__ = SPLICEDense
    return model
