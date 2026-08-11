# Analysis environment for quantum-structure-entanglement.
#
# The base is pinned by DIGEST, not by tag. `nvcr.io/nvidia/pytorch:26.06-py3` is a mutable
# tag; the digest below is immutable and is the thing the reproducibility claim rests on.
# It is duplicated in env/base-image.digest so it can be read without parsing this file.
FROM nvcr.io/nvidia/pytorch@sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1

# The base image supplies python 3.12.3, numpy 2.1.0, scipy 1.17.1, PyYAML 6.0.1 and
# torch 2.13.0a0+8145d630e8.nv26.06. Nothing above is upgraded: upgrading numpy or scipy
# would silently change eigvalsh and the RNG behaviour every committed number depends on.
COPY requirements.lock /tmp/requirements.lock

# Only the two packages the base lacks, plus their pinned transitive closure.
RUN pip install --no-cache-dir --no-deps --root-user-action=ignore \
        scikit-learn==1.9.0 \
        pytest==9.1.1 \
        iniconfig==2.3.0 \
        joblib==1.5.3 \
        narwhals==2.24.0 \
        pluggy==1.6.0 \
        threadpoolctl==3.6.0 \
 && python -c "import sklearn, pytest, joblib, threadpoolctl, numpy, scipy, torch, yaml; print('env ok')"

# The pinned submodule's own runtime dependencies. Needed only because the ms_trained
# checkpoints unpickle into `qsae.reverse_arrow.transformer.TFIMTransformer`, and
# `qsae/__init__.py` eagerly imports `.qnn` -> pennylane. This arm never evaluates a
# quantum circuit; making that import lazy would require editing the pinned submodule,
# which the pin contract forbids. See env/requirements.lock for the reasoning.
RUN pip install --no-cache-dir --no-deps --root-user-action=ignore \
        pennylane==0.45.1 \
        pennylane_lightning==0.45.0 \
        appdirs==1.4.4 \
        autograd==1.8.0 \
        autoray==0.8.4 \
        cachetools==7.1.7 \
        diastatic-malt==2.15.3 \
        rustworkx==0.18.1 \
        scipy-openblas32==0.3.34.106.0 \
        termcolor==3.3.0 \
        tomlkit==0.15.1 \
 && python -c "import pennylane; print('pennylane ok', pennylane.__version__)"

# Dense eigvalsh at L=12 is 4096x4096. Unbounded BLAS threads drive load average past 50 on
# a 20-core box and the suite appears to hang (RESULTS_STAGE1.md runbook note). Bounded here
# so the default is safe; override at `docker run` if a caller really wants more.
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /work

# `qsae` is made importable from the pinned submodule via PYTHONPATH in env/run.sh rather
# than `pip install -e`, so nothing is written into the submodule working tree and the
# gitlink pin stays the sole authority on which code is in use. The ms_trained checkpoints
# are pickles referencing qsae.reverse_arrow.transformer, so torch.load needs it on the path.
