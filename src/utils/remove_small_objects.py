"""Remove instâncias pequenas de um label map e renumeradas as restantes."""

import numpy as np


def remove_small_objects(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Remove instâncias com menos de `min_area` pixels e renumeradas de 1 a N.

    Args:
        labels: label map de instâncias.
        min_area: área mínima, em pixels, para uma instância sobreviver.

    Returns:
        Label map com instâncias pequenas removidas e as restantes renumeradas
        de forma contígua.
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    areas = {label: int(np.sum(labels == label)) for label in unique_labels}
    labels_to_remove = [label for label, area in areas.items() if area < min_area]

    result = labels.copy()
    for label in labels_to_remove:
        result[result == label] = 0

    remaining = np.unique(result)
    remaining = remaining[remaining != 0]
    for new_label, old_label in enumerate(remaining, start=1):
        result[result == old_label] = new_label

    return result
