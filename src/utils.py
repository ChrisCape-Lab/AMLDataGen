import pandas as pd

from itertools import islice
from random import randint

from _constants import GENERAL, SCHEDULING


def add_to_dict_of_list(dictionary: dict, dict_key, dict_item) -> None:
    if dict_key in dictionary.keys():
        dictionary[dict_key].append(dict_item)
    else:
        dictionary[dict_key] = [dict_item]


def addn_to_dict_of_list(dictionary: dict, dict_key, dict_items: list) -> None:
    if dict_key in dictionary.keys():
        dictionary[dict_key].extend(dict_items)
    else:
        dictionary[dict_key] = dict_items


def random_chunk(li, min_chunk=1, max_chunk=3):
    it = iter(li)
    while True:
        nxt = list(islice(it, randint(min_chunk, max_chunk)))
        if nxt:
            yield nxt
        else:
            break


# LOADING UTILS
# ------------------------------------------

def scheduling_string_to_const(scheduling_str: str) -> int:
    if scheduling_str == 'Random':
        return SCHEDULING.RANDOM
    elif scheduling_str == 'Periodic':
        return SCHEDULING.PERIODIC
    elif scheduling_str == 'Instant':
        return SCHEDULING.INSTANT
    else:
        raise NotImplementedError


def pattern_string_to_const(pattern_str: str) -> int:
    if pattern_str == 'Fan_in':
        return GENERAL.FAN_IN
    elif pattern_str == 'Fan_out':
        return GENERAL.FAN_OUT
    elif pattern_str == 'Cycle':
        return GENERAL.CYCLE
    elif pattern_str == 'Scatter-Gather':
        return GENERAL.SCATTER_GATHER
    elif pattern_str == 'Gather-Scatter':
        return GENERAL.GATHER_SCATTER
    elif pattern_str == 'U_Pattern':
        return GENERAL.U
    elif pattern_str == 'Repeated':
        return GENERAL.REPEATED
    elif pattern_str == 'Bipartite':
        return GENERAL.BIPARTITE

def get_degrees(deg_csv, num_v):
    """
    :param deg_csv: Degree distribution parameter CSV file
    :param num_v: Number of total account vertices
    :return: In-degree and out-degree sequence list
    """
    deg_df = pd.read_csv(deg_csv)

    _in_deg = list()  # In-degree sequence
    _out_deg = list()  # Out-degree sequence

    for row in deg_df.iterrows():
        count = int(row[0])
        _in_deg.extend([int(row[1])] * count)
        _out_deg.extend([int(row[2])] * count)

    in_len, out_len = len(_in_deg), len(_out_deg)
    if in_len != out_len:
        raise ValueError("The length of in-degree (%d) and out-degree (%d) sequences must be same."
                         % (in_len, out_len))

    total_in_deg, total_out_deg = sum(_in_deg), sum(_out_deg)
    if total_in_deg != total_out_deg:
        raise ValueError("The sum of in-degree (%d) and out-degree (%d) must be same."
                         % (total_in_deg, total_out_deg))

    if num_v % in_len != 0:
        raise ValueError("The number of total accounts (%d) "
                         "must be a multiple of the degree sequence length (%d)."
                         % (num_v, in_len))

    repeats = num_v // in_len
    _in_deg = _in_deg * repeats
    _out_deg = _out_deg * repeats

    return _in_deg, _out_deg


def add_edge_id(self, orig, bene):
    """Adds info to edge. Based on add_transaction. Add transaction will go away eventually.
    :param orig: Originator account ID
    :param bene: Beneficiary account ID
    :return:
    """
    if orig == bene:
        raise ValueError("Self loop from/to %s is not allowed for transaction networks" % str(orig))

