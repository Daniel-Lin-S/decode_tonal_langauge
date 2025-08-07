import numpy as np
from sklearn import metrics as skmetrics


def compute_classification_metrics_joint(
        all_true: dict, all_preds: dict,
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
    metrics : list, optional
        A list of metric names to compute. Names should correspond
        to functions in ``sklearn.metrics`` (e.g. 'accuracy',
        'f1', 'precision', 'recall', 'cohen_kappa',
        'confusion_matrix'). Default is ['accuracy'].

    Returns
    -------
    dict
        A dictionary where keys are metric names and
        values are the computed metric values.
    """
    if set(all_true.keys()) != set(all_preds.keys()):
        raise ValueError("Keys in all_true and all_preds must match.")

    targets = list(all_true.keys())

    results = {}

    for target in targets:
        print('Unique labels in {}: {}'.format(
            target, set(all_true[target])
        ))
        print('Unique predictions in {}: {}'.format(
            target, set(all_preds[target])
        ))

    all_true = {target: all_true[target].astype(int) for target in targets}
    all_preds = {target: all_preds[target].astype(int) for target in targets}

    # Flatten the joint true and predicted labels into single numeric arrays
    n_classes = [len(np.unique(all_true[target])) for target in targets]

    multipliers = np.array([np.prod(n_classes[i + 1:]) for i in range(len(n_classes))])

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

    if verbose:
        print('Unique labels in joint truth: {}'.format(
            set(joint_true)
        ), flush=True)
        print(
            'Unique labels in joint predictions: {}'.format(set(joint_preds))
            , flush=True)

    metric_funcs = {
        'accuracy': skmetrics.accuracy_score,
        'f1': lambda y_true, y_pred: skmetrics.f1_score(y_true, y_pred, average='weighted'),
        'precision': lambda y_true, y_pred: skmetrics.precision_score(y_true, y_pred, average='weighted'),
        'recall': lambda y_true, y_pred: skmetrics.recall_score(y_true, y_pred, average='weighted'),
        'cohen_kappa': skmetrics.cohen_kappa_score,
        'confusion_matrix': skmetrics.confusion_matrix
    }

    for m in metrics:
        if m in metric_funcs:
            results[m] = metric_funcs[m](joint_true, joint_preds)
        else:
            try:
                results[m] = getattr(skmetrics, m)(joint_true, joint_preds)
            except AttributeError:
                raise ValueError(
                    f"Metric '{m}' is not recognized in sklearn.metrics, and "
                    f"not part of the supported metrics: {list(metric_funcs.keys())}."
                )

    return results
