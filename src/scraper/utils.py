import re
from typing import Optional, Tuple

def clean_runs(runs_str: str) -> Tuple[int, int]:
    """
    Cleans the Runs string, extracting numeric value and Not Out status.
    Handles 'DNB', 'TDNB', 'absent', 'sub', '-' as 0.
    """
    if not runs_str or runs_str.lower() in ['dnb', 'tdnb', 'absent', 'sub', '-']:
        return 0, 0
    
    not_out = 1 if '*' in runs_str else 0
    clean_val = runs_str.replace('*', '').strip()
    try:
        return int(clean_val), not_out
    except ValueError:
        return 0, not_out

def clean_generic_stat(stat_str: str) -> float:
    """
    Converts a generic stat string to float, handling '-', 'DNB', etc. as 0.
    """
    if not stat_str or stat_str.strip() in ['-', 'DNB', 'TDNB', 'absent', 'sub']:
        return 0.0
    try:
        return float(stat_str)
    except ValueError:
        return 0.0

def extract_player_info(player_str: str) -> Tuple[str, str]:
    """
    Extracts Player Name and Country from a string like 'Mohammad Ihsan (ESP)'.
    """
    match = re.search(r'^(.*?) \((.*?)\)', player_str)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return player_str.strip(), ""

def clean_opposition(opp_str: str) -> str:
    """
    Removes the 'v ' prefix from the opposition string.
    """
    return opp_str.lstrip('v ').strip()

def replace_special_values(value):
    """
    Generic replacer for DNB and other non-numeric markers.
    Used for dataframe mapping if needed.
    """
    special = ['DNB', 'TDNB', 'absent', 'sub', '-']
    if str(value).strip() in special:
        return 0
    return value
