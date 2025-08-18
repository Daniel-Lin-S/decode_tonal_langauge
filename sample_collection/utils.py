import re
from typing import List, Optional


def extract_block_id(filename: str) -> int:
    """
    Extract the integer block ID from a filename containing 'B{block_id}'.
    
    Parameters
    ----------
    filename : str
        The file name (e.g., 'HS25_B1.wav', 'B5_16000.TextGrid')
    
    Returns
    -------
    int
        Extracted block ID as integer
    
    Raises
    ------
    ValueError
        If no 'B<number>' pattern is found.
    """
    match = re.search(r'B(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"No block ID found in filename: {filename}")

def match_filename(
        file: str, file_format: str,
        kwords: Optional[List[str]]=None,
    ) -> bool:
    """
    Check if the file is of desired form and
    have all the keywords in the filename.

    Parameters
    ----------
    file : str
        The filename to check.
    file_format : str
        The expected file format (e.g., 'wav', 'TextGrid').
    kwords : List[str], optional
        List of keywords to check in the filename.
        If None, no keywords are checked.

    Returns
    -------
    bool
        True if the file matches the criteria, False otherwise.
    """
    # Check file extension
    if not file.endswith(file_format):
        return False

    condition = True

    if kwords is not None:
        for word in kwords:
            if word not in file:
                condition = False

    return condition
