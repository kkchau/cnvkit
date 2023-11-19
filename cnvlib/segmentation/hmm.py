"""Segmentation by Hidden Markov Model."""
import collections
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import scipy.special

from cnvlib.descriptives import biweight_midvariance
from skgenome.gary import GenomicArray

from ..cnary import CopyNumArray as CNA
from ..segfilters import squash_by_groups

if TYPE_CHECKING:
    from pomegranate.distributions import Normal
    from pomegranate.hmm import DenseHMM
else:
    Normal = object
    DenseHMM = object

logger = logging.getLogger(__name__)

DISTRIBUTION_INERTIA = 0.8  # Allow updating distributions, but slowly
EDGE_INERTIA = 0.1


def segment_hmm(
    cnarr: CNA,
    method: str,
    diploid_parx_genome: Optional[str] = None,
    window: Optional[float] = None,
) -> GenomicArray:
    """Segment bins by Hidden Markov Model.

    Use Viterbi method to infer copy number segments from sequential data.

    With b-allele frequencies ('baf' column in `cnarr`), jointly segment
    log-ratios and b-allele frequencies across a chromosome.

    Parameters
    ----------
    cnarr : CopyNumArray
        The bin-level data to segment.
    method : string
        One of 'hmm' (3 states, flexible means), 'hmm-tumor' (5 states, flexible
        means), 'hmm-germline' (3 states, fixed means).
    diploid_parx_genome : string
        Whether to include PAR1/2 from chr X within the autosomes.

    Results
    -------
    segarr : CopyNumArray
        The segmented data.
    """
    # NB: Incorporate weights into smoothed log2 estimates
    # (Useful kludge until weighted HMM is in place)
    orig_log2 = cnarr["log2"].values.copy()
    cnarr["log2"] = cnarr.smooth_log2(window)

    logger.info("Building model from observations")
    model = hmm_get_model(cnarr, method, diploid_parx_genome)

    logger.info("Predicting states from model")
    observations = np.array(
        [[[x] for x in np.concatenate(as_observation_matrix(cnarr))]]
    )
    states = model.predict(observations)

    logger.info("Done, now finalizing")
    logger.debug("Predicted states: %s", states)
    logger.debug(str(collections.Counter([state.item() for state in states[0]])))
    logger.debug(
        "Observations: %s", [observation[0] for observation in observations[0][:100]]
    )
    logger.debug("Edges: %s", model.edges)

    # Merge adjacent bins with the same state to create segments
    cnarr["log2"] = orig_log2
    cnarr["probes"] = 1
    segarr = squash_by_groups(
        cnarr, pd.Series(states[0], index=cnarr.data.index), by_arm=True
    )
    if not (segarr.start < segarr.end).all():
        bad_segs = segarr[segarr.start >= segarr.end]
        logger.warning("Bad segments:\n%s", bad_segs.data)
    return segarr


def hmm_get_model(
    cnarr: CNA, method: str, diploid_parx_genome: Optional[str] = None
) -> DenseHMM:
    """

    Parameters
    ----------
    cnarr : CopyNumArray
        The normalized bin-level values to be segmented.
    method : string
        One of 'hmm' (3 states, flexible means), 'hmm-tumor' (5 states, flexible
        means), 'hmm-germline' (3 states, fixed means).
    diploid_parx_genome : string
        Whether to include PAR1/2 from chr X within the autosomes.

    Returns
    -------
    model :
        A pomegranate HiddenMarkovModel trained on the given dataset.
    """
    try:
        from pomegranate.hmm import DenseHMM
    except ImportError:
        logger.exception(
            "Calling with hmm requires pomegranate. Reinstall with hmm support via `cnvkit[hmm]`."
        )
        raise
    assert (
        method in ("hmm-tumor", "hmm-germline", "hmm")
    ), f"Invalid method {method} specified. Expected one of 'hmm-tumor', 'hmm-germline', or 'hmm'."
    observations = np.concatenate(
        as_observation_matrix(cnarr.autosomes(diploid_parx_genome=diploid_parx_genome))
    )
    variance = biweight_midvariance(observations) ** 2

    if method == "hmm-germline":
        state_names = ["loss", "neutral", "gain"]
        distributions = [
            normal_distribution(mean=-1.0, variance=variance, frozen=True),
            normal_distribution(mean=0.0, variance=variance, frozen=True),
            normal_distribution(mean=0.585, variance=variance, frozen=True),
        ]
    elif method == "hmm-tumor":
        state_names = ["del", "loss", "neutral", "gain", "amp"]
        distributions = [
            normal_distribution(mean=-2.0, variance=variance, frozen=False),
            normal_distribution(mean=-0.5, variance=variance, frozen=False),
            normal_distribution(mean=0.0, variance=variance, frozen=True),
            normal_distribution(mean=0.3, variance=variance, frozen=False),
            normal_distribution(mean=1.0, variance=variance, frozen=False),
        ]
    else:
        state_names = ["loss", "neutral", "gain"]
        distributions = [
            normal_distribution(mean=-1.0, variance=variance, frozen=False),
            normal_distribution(mean=0.0, variance=variance, frozen=False),
            normal_distribution(mean=0.585, variance=variance, frozen=False),
        ]

    n_states = len(distributions)
    # Starts -- prefer neutral
    binom_coefs = scipy.special.binom(n_states - 1, range(n_states))
    start_probabilities = binom_coefs / binom_coefs.sum()

    # Prefer to keep the current state in each transition
    # All other transitions are equally likely, to start
    transition_matrix = (
        (np.identity(n_states, dtype=np.float32) * 100)
        + (np.ones((n_states, n_states), dtype=np.float32) / n_states)
    ) / (100 + (1 / n_states))

    model = DenseHMM(
        distributions=distributions,
        edges=transition_matrix,
        starts=start_probabilities,
        inertia=0.1,
    )
    X = np.array([[[x] for x in observations]])

    # Fit the model to the observations
    model.fit(X=X)

    return model


def as_observation_matrix(cnarr, variants=None):
    """Extract HMM fitting values from `cnarr`.

    For each chromosome arm, extract log2 ratios as a numpy array.

    Future: If VCF of variants is given, or 'baf' column has already been
    added to `cnarr` from the same, then the BAF values are a second row/column
    in each numpy array.

    Returns: List of numpy.ndarray, one per chromosome arm.
    """
    # TODO incorporate weights -- currently handled by smoothing
    # TODO incorporate inter-bin distances
    observations = [arm.log2.values for _c, arm in cnarr.by_arm()]
    return observations


def normal_distribution(mean: float, variance: float, frozen: bool) -> Normal:
    try:
        from pomegranate.distributions import Normal
    except ImportError:
        logger.exception(
            "Calling with hmm requires pomegranate. Reinstall with hmm support via `cnvkit[hmm]`."
        )
        raise
    else:
        return Normal(
            means=[mean],
            covs=[[np.float32(variance)]],
            covariance_type="full",
            inertia=DISTRIBUTION_INERTIA,
            frozen=frozen,
        )


def variants_in_segment(varr, segment, min_variants=50):
    try:
        from pomegranate.hmm import DenseHMM
    except ImportError:
        logger.exception(
            "Calling with hmm requires pomegranate. Reinstall with hmm support via `cnvkit[hmm]`."
        )
        raise
    if len(varr) > min_variants:
        observations = varr.mirrored_baf(above_half=True)
        state_names = ["neutral", "alt"]
        distributions = [
            normal_distribution(0.5, 0.1, frozen=True),
            normal_distribution(0.67, 0.1, frozen=True),
        ]
        n_states = len(distributions)
        # Starts -- prefer neutral
        start_probabilities = [0.95, 0.05]
        # Prefer to keep the current state in each transition
        # All other transitions are equally likely, to start
        transition_matrix = (
            (np.identity(n_states, dtype=np.float32) * 100)
            + (np.ones((n_states, n_states), dtype=np.float32) / n_states)
        ) / (100 + (1 / n_states))

        model = DenseHMM(
            distributions=distributions,
            edges=transition_matrix,
            starts=start_probabilities,
            inertia=0.1,
        )

        X = np.array([[[x] for x in observations]])
        model.fit(X=X)
        states = np.array(model.predict(X))

        logger.info("Done, now finalizing")
        logger.debug("Predicted states: %s", states)
        logger.debug(str(collections.Counter([state.item() for state in states[0]])))
        logger.debug(
            "Observations: %s",
            [observation[0] for observation in observations[0][:100]],
        )
        logger.debug("Edges: %s", model.edges)

        # Merge adjacent bins with the same state to create segments
        fake_cnarr = CNA(varr.add_columns(weight=1, log2=0, gene=".").data)
        results = squash_by_groups(fake_cnarr, varr.as_series(states), by_arm=False)
        assert (results.start < results.end).all()

    else:
        results = None

    if results is not None and len(results) > 1:
        logger.info(
            "Segment %s:%d-%d on allele freqs for %d additional breakpoints",
            segment.chromosome,
            segment.start,
            segment.end,
            len(results) - 1,
        )
        # Place breakpoints midway between SNVs
        # XXX TODO use original cnarr bin boundaries to select/adjust breakpoint
        mid_breakpoints = (results.start.values[1:] + results.end.values[:-1]) // 2
        starts = np.concatenate([[segment.start], mid_breakpoints])
        ends = np.concatenate([mid_breakpoints, [segment.end]])
        dframe = pd.DataFrame(
            {
                "chromosome": segment.chromosome,
                "start": starts,
                "end": ends,
                # 'baf': results['mean'],
                "gene": segment.gene,  # '-'
                "log2": segment.log2,
                "probes": results["probes"],
                # 'weight': (segment.weight * results['probes']
                #            / (segment.end - segment.start)),
            }
        )
        bad_segs_idx = dframe.start >= dframe.end
        if bad_segs_idx.any():
            raise RuntimeError(
                f"Improper post-processing of segment {segment} -- "
                f"{bad_segs_idx.sum()} bins start >= end:\n{dframe[bad_segs_idx]}\n"
            )

    else:
        dframe = pd.DataFrame(
            {
                "chromosome": segment.chromosome,
                "start": segment.start,
                "end": segment.end,
                "gene": segment.gene,  # '-',
                "log2": segment.log2,
                "probes": segment.probes,
                # 'weight': segment.weight,
            },
            index=[0],
        )

    return dframe
