"""
Ball-number parser for Cricsheet data.

Cricsheet encodes the delivery information in a single ``ball`` column using
a decimal format (e.g. ``5.3`` means over 5, ball 3).  This module separates
that into two integer columns: ``completed_over`` and ``ball_no``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def separate_ball_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Split the ``ball`` column into ``completed_over`` and ``ball_no``.

    The original ``ball`` column is retained for reference but the two new
    columns are appended to *df* **in-place** (a copy is *not* made).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``ball`` column with numeric-like values.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with two new integer columns added.

    Notes
    -----
    The legacy notebook used the following heuristic:

    * Multiply the ``ball`` value by 10.
    * If the result is between 10 and 100 (inclusive), use ``// 100`` and
      ``% 100`` to extract over and ball.
    * Otherwise use ``// 10`` and ``% 10``.

    Non-numeric values are mapped to ``NaN``.
    """
    completed_overs: list[int | float] = []
    ball_nos: list[int | float] = []

    for ball in df["ball"]:
        try:
            ball_num = float(ball) * 10
            if 10 <= ball_num <= 100:
                completed_overs.append(int(ball_num // 100))
                ball_nos.append(int(ball_num % 100))
            else:
                completed_overs.append(int(ball_num // 10))
                ball_nos.append(int(ball_num % 10))
        except (ValueError, TypeError):
            completed_overs.append(np.nan)
            ball_nos.append(np.nan)

    df["completed_over"] = completed_overs
    df["ball_no"] = ball_nos
    return df
