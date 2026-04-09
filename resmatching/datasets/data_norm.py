"""Per-dataset, per-channel normalisation statistics for BioSR.

Each entry maps a dataset key to ([mean_ch0, mean_ch1], [std_ch0, std_ch1]).
Channel 0 = low-res (noisy) input; channel 1 = high-res (clean) target.
"""

STATS: dict[str, tuple[list[float], list[float]]] = {
    "ccp": ([310.15253, 102.429474], [2256.7798, 3.7435353]),
    "er": ([7221.413, 101.77709], [6812.191, 4.962002]),
    "factin": ([6608.2, 112.05498], [5541.5005, 10.140036]),
    "mt": ([5753.911, 110.81242], [5985.273, 9.165207]),
    "mt_noisy": ([5753.911, 110.81217], [5985.273, 22.127481]),
}


def normalize(image, dataset: str, channel: int):
    mean, std = STATS[dataset]
    return (image - mean[channel]) / std[channel]


def denormalize(image, dataset: str, channel: int):
    mean, std = STATS[dataset]
    return image * std[channel] + mean[channel]
