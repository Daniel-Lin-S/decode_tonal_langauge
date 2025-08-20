import pandas as pd
import numpy as np
from textgrid import TextGrid, Interval
import os
from typing import List, Optional, Dict
import warnings
from argparse import Namespace

from .utils import extract_block_id


def get_intervals(
        params: Namespace
    ) -> Dict[int, pd.DataFrame]:
    """
    Extracts information from TextGrid files in the specified directory.
    Each TextGrid file must have a naming convention that includes
    a block number at the end,
    e.g. 'example_TextGrid_B1.TextGrid'.
    Each block will only be loaded once.

    Parameters
    ----------
    params : Namespace
        Parameters for the extraction process.
        Should contain:
        - textgrid_dir : str, directory containing TextGrid files.
        - start_offset (optional) : float, offset to subtract from the
          start time of each interval, in seconds. Defaults to 0.0.
        - end_offset (optional) : float, offset to add to the end time of each interval,
          in seconds. Defaults to 0.0.
        - tier_list (optional) : List[str]
          List of tier names to extract from the TextGrid.
          If None, all tiers will be considered.
        - blocks (optional) : List[int]
          List of block numbers to process.
          If None, all blocks will be processed.
        - dataset (optional) : str, dataset name to determine the extraction method.
            Supported values are 'liu2023' and 'zhang2024'.
            Defaults to 'liu2023'.
    
    Returns
    -------
    Dict[int, pd.DataFrame]
        A dictionary where keys are block numbers and
        values are DataFrames containing the extracted intervals
        with columns:
        - start : float, start time of the interval (in seconds)
        - end : float, end time of the interval (in seconds)
        - syllable : str, syllable associated with the interval
        - tone : int, tone associated with the interval
    """
    dataset = getattr(params, 'dataset', 'liu2023')
    blocks = getattr(params, 'blocks', None)
    tier_list = getattr(params, 'tier_list', None)
    start_offset = getattr(params, 'start_offset', 0.0)
    end_offset = getattr(params, 'end_offset', 0.0)

    intervals = {}

    for file in os.listdir(params.textgrid_dir):
        if file.endswith('.TextGrid'):
            block_number = extract_block_id(file)
            
            if blocks is not None and block_number not in blocks:
                continue

            if block_number not in intervals:
                file_path = os.path.join(params.textgrid_dir, file)
                tg = TextGrid.fromFile(file_path)

                block_data = read_textgrid(
                    tg, start_offset, end_offset, tier_list,
                    dataset=dataset)

                if len(block_data) == 0:
                    raise ValueError(
                        f"No valid intervals found in TextGrid file: {file_path}. "
                        "Please check the file format and content."
                    )

                total_len = get_textgrid_time(tg, tier_list)
                print(
                    f'Maximum time for block {block_number}:',
                    total_len, ' s',
                    flush=True
                )

                intervals[block_number] = block_data

    return intervals


def read_textgrid(
        tg : TextGrid,
        start_offset: float, end_offset: float,
        tier_list: Optional[List[str]]=None,
        dataset: str='liu2023'
    ) -> pd.DataFrame:
    """
    Find the intervals with marks starting with a digit
    (indicates true reading times) and extract them into a DataFrame.
    Required textgrid format:
    <tone><syllable>, e.g. '3ma', '4shi', etc.

    Parameters
    ----------
    tg : TextGrid
        TextGrid object containing the intervals.
    start_offset : float
        Offset to subtract from the start time of each interval,
        in seconds
    end_offset : float
        Offset to add to the end time of each interval,
        in seconds
    tier_list : List[str], optional
        List of tier names to extract from the TextGrid.
        If None, all tiers will be considered.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the extracted intervals.
        Each row represents an interval with the following columns:
        - start : float, start time of the interval (in seconds)
        - end : float, end time of the interval (in seconds)
        - syllable : str, syllable associated with the interval
        - tone : int, tone associated with the interval
    """

    tier_names = [tier.name for tier in tg.tiers]

    if tier_list is None:
        tier_list = tier_names
    else:
        for tier in tier_list:
            if tier not in tier_names:
                raise ValueError(
                    f"Tier '{tier}' not found in TextGrid. "
                    f"Available tiers: {tier_names}."
                )

    extract_interval_func = {
        'liu2023': extract_interval_syllable_tone_liu2023,
        'zhang2024': extract_interval_syllable_tone_zhang2024,
    }.get(dataset)

    if extract_interval_func is None:
        raise ValueError(f"Unsupported dataset: {dataset}")

    trial_list = [
        interval_data
        for tier in tg.tiers if tier.name in tier_list
        for interval in tier.intervals
        if (interval_data := extract_interval_func(
            interval, start_offset, end_offset)) is not None
    ]

    return pd.DataFrame(trial_list)


def extract_interval_syllable_tone_liu2023(
    interval: Interval,
    start_offset: float, end_offset: float
) -> Optional[Dict[str, float]]:
    """
    Extracts syllable and tone from an interval in the format
    <tone><syllable>, e.g. '3ma', '4shi', etc.
    
    Parameters
    ----------
    interval : Interval
        The interval from which to extract syllable and tone.
    start_offset : float
        Offset to subtract from the start time of the interval,
        in seconds.
    end_offset : float
        Offset to add to the end time of the interval,
        in seconds.

    Returns
    -------
    Optional[Dict[str, float]]
        A dictionary with keys 'start', 'end', 'syllable', and 'tone',
        or None if the interval does not match the expected format.

    Reference
    ----------
    This format is used in the dataset of Yan Liu et al. ,
    Decoding and synthesizing tonal language speech from brain activity.
    Sci. Adv.9,eadh0478(2023).DOI:10.1126/sciadv.adh0478
    """
    if len(interval.mark) == 0 or not interval.mark[0].isdigit():
        return None

    tone = int(interval.mark[0])
    syllable = interval.mark[1:]

    start = interval.minTime - start_offset
    end = interval.maxTime + end_offset

    return {
        'start': np.around(start, decimals=1),
        'end': np.around(end, decimals=1),
        'syllable': syllable,
        'tone': tone
    }

def extract_interval_syllable_tone_zhang2024(
    interval: Interval,
    start_offset: float, end_offset: float,
    allowed_tones: List[int]=[1, 2, 3, 4]
) -> Optional[Dict[str, float]]:
    """
    Extracts syllable and tone from an interval in the format
    <syllable><tone>, e.g. 'ma3', 'shi4', etc.
    
    Parameters
    ----------
    interval : Interval
        The interval from which to extract syllable and tone.
    start_offset : float
        Offset to subtract from the start time of the interval,
        in seconds.
    end_offset : float
        Offset to add to the end time of the interval,
        in seconds.

    Returns
    -------
    Optional[Dict[str, float]]
        A dictionary with keys 'start', 'end', 'syllable', and 'tone',
        or None if the interval does not match the expected format.

    Reference
    ----------
    This format is used in the paper Daohan Zhang, Zhenjie Wang, Youkun Qian, Zehao Zhao,
    Yan Liu, Xiaotao Hao, Wanxin Li, Shuo Lu, Honglin Zhu, Luyao Chen, Kunyu Xu, Yuanning Li, Junfeng Lu,
    A brain-to-text framework for decoding natural tonal sentences,
    Cell Reports,
    Volume 43, Issue 11, 2024, 114924, ISSN 2211-1247,
    https://doi.org/10.1016/j.celrep.2024.114924.
    """
    if len(interval.mark) == 0 or not any(char.isdigit() for char in interval.mark):
        return None

    tone = ''.join([char for char in interval.mark if char.isdigit()])
    if int(tone) not in allowed_tones:
        warnings.warn(
            f"Unexpected tone '{tone}' found in interval mark '{interval.mark}'. "
            f"Allowed tones are {allowed_tones}. Skipping this interval."
        )
        return None

    if not tone:
        return None

    digit_index = interval.mark.index(tone[0])
    syllable = interval.mark[:digit_index]

    start = interval.minTime - start_offset
    end = interval.maxTime + end_offset

    return {
        'start': np.around(start, decimals=1),
        'end': np.around(end, decimals=1),
        'syllable': syllable,
        'tone': int(tone)
    }


def get_textgrid_time(
        tg: TextGrid,
        tier_list: Optional[List[str]]=None
    ) -> float:
    """
    Calculates the total temporal length of the TextGrid file
    based on the specified tiers.

    Parameters
    ----------
    tg : TextGrid
        TextGrid object containing the intervals.
    tier_list : List[str], optional
        List of tier names to consider for length calculation.
        If None, all tiers will be considered.

    Returns
    -------
    float
        The maximum end time of the intervals in the specified tiers.
    """
    if tier_list is None:
        tier_list = [tier.name for tier in tg.tiers]

    max_time = 0.0

    for tier in tg.tiers:
        if tier.name in tier_list:
            for interval in tier.intervals:
                if interval.maxTime > max_time:
                    max_time = interval.maxTime

    return max_time
