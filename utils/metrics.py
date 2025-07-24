import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
)


def compute_joint_metrics(
        all_true: dict, all_preds: dict,
        metrics: list = ['accuracy']
    ) -> dict:
    """
    Compute joint metrics (e.g., accuracy, F1 score)
    based on true and predicted labels for multiple targets.
    Only suitable for classification tasks.

    Parameters
    ----------
    all_true : dict
        A dictionary where keys are target variable names
        and values are arrays of true labels.
    all_preds : dict
        A dictionary where keys are target variable names
        and values are arrays of predicted labels.
    metrics : list, optional
        A list of metrics to compute. \n
        Supported metrics: ['accuracy', 'f1', 'cohen's kappa'].
        Default is ['accuracy'].

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
    print('Unique labels in joint truth: {}'.format(
        set(joint_true)
    ), flush=True)
    print(
        'Unique labels in joint predictions: {}'.format(set(joint_preds))
        , flush=True)

    if 'accuracy' in metrics:
        results['accuracy'] = accuracy_score(joint_true, joint_preds)

    if 'f1_score' in metrics:
        results['f1_score'] = f1_score(
            joint_true, joint_preds,
            average='weighted'
        )

    if 'cohen_kappa' in metrics:
        results['cohen_kappa'] = cohen_kappa_score(
            joint_true, joint_preds
        )

    if 'confusion_matrix' in metrics:
        results['confusion_matrix'] = confusion_matrix(
            joint_true, joint_preds
        )

        print('Shape of confusion matrix: {}'.format(
            results['confusion_matrix'].shape,
            flush=True
        ))

    return results