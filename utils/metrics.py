import numpy as np
from sklearn import metrics as skmetrics


def compute_classification_metrics(
        true: np.ndarray, preds: np.ndarray,
        n_classes: int,
        metrics: list = ['accuracy'],
        verbose: bool = False
    ) -> dict:
    """
    Compute metrics (e.g., accuracy, F1 score)
    for a single classification task based on true and predicted labels.

    Parameters
    ----------
    true : np.ndarray
        Array of true labels.
    preds : np.ndarray
        Array of predicted labels.
    metrics : list, optional
        A list of metric names to compute. Names should correspond
        to functions in ``sklearn.metrics`` (e.g. 'accuracy',
        'f1', 'precision', 'recall', 'cohen_kappa',
        'confusion_matrix'). Default is ['accuracy'].
    verbose : bool, optional
        If True, prints unique labels and predictions.

    Returns
    -------
    dict
        A dictionary where keys are metric names and
        values are the computed metric values.
    """
    if not isinstance(n_classes, int):
        raise ValueError(
            "n_classes must be an integer. "
            f"Got {n_classes} of type {type(n_classes)}. "
        )

    if verbose:
        print('Unique labels in true: {}'.format(set(true)))
        print('Unique predictions in preds: {}'.format(set(preds)))

    # ensure the shape of confusion matrix is consistent
    all_labels = np.arange(n_classes)

    metric_funcs = {
        'accuracy': skmetrics.accuracy_score,
        'f1_score': lambda y_true, y_pred: skmetrics.f1_score(
            y_true, y_pred, average='weighted'),
        'precision': lambda y_true, y_pred: skmetrics.precision_score(
            y_true, y_pred, average='weighted'),
        'recall': lambda y_true, y_pred: skmetrics.recall_score(
            y_true, y_pred, average='weighted'),
        'cohen_kappa': skmetrics.cohen_kappa_score,
        'confusion_matrix': lambda y_true, y_pred: skmetrics.confusion_matrix(
            y_true, y_pred, labels=all_labels),
    }

    results = {}
    for m in metrics:
        if m in metric_funcs:
            results[m] = metric_funcs[m](true, preds)
        else:
            try:
                metric_func = getattr(skmetrics, m)
                if 'average' in metric_func.__code__.co_varnames:
                    results[m] = metric_func(true, preds, average='weighted')
                else:
                    results[m] = metric_func(true, preds)
            except AttributeError:
                raise ValueError(
                    f"Metric '{m}' is not recognized in sklearn.metrics, and "
                    f"not part of the supported metrics: {list(metric_funcs.keys())}."
                )

    return results


def compute_classification_metrics_joint(
        all_true: dict, all_preds: dict,
        n_classes: int,
        metrics: list = ['accuracy'],
        verbose: bool = False
    ) -> dict:
    """
    Compute joint metrics (e.g., accuracy, F1 score)
    for classification tasks based on true and predicted labels
    for multiple targets.

    Parameters
    ----------
    all_true : dict
        A dictionary where keys are target variable names
        and values are arrays of true labels.
    all_preds : dict
        A dictionary where keys are target variable names
        and values are arrays of predicted labels.
    n_classes : int
        Total number of classes across all targets.
    metrics : list, optional
        A list of metric names to compute. Names should correspond
        to functions in ``sklearn.metrics`` (e.g. 'accuracy',
        'f1', 'precision', 'recall', 'cohen_kappa',
        'confusion_matrix'). Default is ['accuracy'].
    verbose : bool, optional
        If True, prints unique labels and predictions for each target.
        Default is False.

    Returns
    -------
    dict
        A dictionary where keys are metric names and
        values are the computed metric values.
    """
    if set(all_true.keys()) != set(all_preds.keys()):
        raise ValueError("Keys in all_true and all_preds must match.")

    targets = list(all_true.keys())

    if verbose:
        for target in targets:
            print('Unique labels in {}: {}'.format(
                target, set(all_true[target])
            ))
            print('Unique predictions in {}: {}'.format(
                target, set(all_preds[target])
            ))

    all_true = {target: all_true[target].astype(int) for target in targets}
    all_preds = {target: all_preds[target].astype(int) for target in targets}

    n_classes_list = [len(np.unique(all_true[target])) for target in targets]

    multipliers = np.array([np.prod(n_classes_list[i + 1:]) for i in range(len(n_classes_list))])

    joint_true = np.sum(
        np.stack(
            [all_true[target] * multiplier
            for target, multiplier in zip(targets, multipliers)],
            axis=1
        ),
        axis=1
    )
    joint_preds = np.sum(
        np.stack(
            [all_preds[target] * multiplier
            for target, multiplier in zip(targets, multipliers)],
            axis=1
        ),
        axis=1
    )

    return compute_classification_metrics(
        joint_true, joint_preds, n_classes, metrics, verbose
    )
