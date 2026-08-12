"""Shims for cornac behaviour the runners cannot use as-is.

Nothing here changes what a model learns. These are compatibility fixes for
places where cornac's own code raises or misbehaves under the configuration
this project needs; anything that would alter training, evaluation, or negative
sampling belongs in `timeaware_data.py` instead, where the ablation can see it.
"""

from cornac.models import NeuMF as _CornacNeuMF
from cornac.models.recommender import Recommender


class NeuMF(_CornacNeuMF):
    """cornac's NeuMF, saved the way every other cornac model is.

    Same name as the class it subclasses, deliberately: callers swap
    `from cornac.models import NeuMF` for `from ..lib.cornac_compat import NeuMF`
    and nothing else changes. This *is* cornac's NeuMF for every purpose the
    ablation touches, and the import path is where the difference belongs.

    `NCFBase.save()` writes the pickle first and only then trips over its own
    unfinished business::

        model_file = Recommender.save(self, save_dir)   # the .pkl, written fine
        if self.backend == "tensorflow":
            self.model.save_weights(model_file.replace(".pkl", ".h5"))
        elif self.backend == "pytorch":
            raise NotImplementedError()                 # TODO, still open in 2.6.0

    The unimplemented part is only the *separate backend weight export* -- the
    `.h5` sidecar the TensorFlow path writes. cornac raises rather than skipping
    it, and since `Experiment.run()` calls `save()` whenever `save_dir` is set,
    a plain NeuMF trains, evaluates, writes its pickle, and *then* aborts,
    taking the metrics down with it before the runner can checkpoint them.

    So this skips exactly the raise and nothing else, going straight to
    `Recommender.save` for the same `.pkl` + `.meta` any other cornac model
    produces. The pickle holds the torch module and round-trips, so nothing is
    lost by not writing the sidecar. Training, evaluation, and negative sampling
    are untouched.

    GMF and MLP inherit the same `NCFBase.save`, so they will need the same
    treatment if either is ever added to the ablation.
    """

    def save(self, save_dir=None, *args, **kwargs):
        if save_dir is None:
            return None
        return Recommender.save(self, save_dir, *args, **kwargs)
