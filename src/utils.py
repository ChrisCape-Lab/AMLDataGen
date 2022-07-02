from threading import Thread

import pandas as pd
from multiprocessing import Process, Manager
from time import sleep

from itertools import islice
from random import randint


class ThreadWithReturnValue(Thread):
    def __int__(self, group=None, target=None, name=None, args=(), kwargs=(), Verbose=None):
        Thread.__init__(group, target, name, args, kwargs, Verbose)
        self._return = None

        def run(self) -> None:
            print(type(self._target))
            if self._target is not None:
                self._return = self._target(*self._args, **self._kwargs)

        def join(self, *args):
            Thread.join(self, *args)
            return self._return


class NodeRequirements:
    def __init__(self, node_id):
        self.__node_id_to_map = node_id
        self.__node_requirements = dict()

    def get_node_id_to_map(self):
        return self.__node_id_to_map

    def get_requirement_value(self, key: str):
        _, value = self.__node_requirements[key]
        return value

    def get_requirement_as_condition(self, key: str) -> (str, ):
        sign, value = self.__node_requirements[key]
        requirement = key + " " + sign + " " + str(value)
        return requirement

    def get_requirements_keys(self):
        return self.__node_requirements.keys()

    def add_requirement(self, key: str, sign: str, value) -> None:
        self.__node_requirements[key] = (sign, value)


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


def integer_amount_partition(amount: float, number_batch: int):
    q = round(amount / number_batch, 2)
    r = round(amount - q * number_batch, 2)

    return q, r


def run_with_limited_time(function, args, time):

    def return_handler_function(function, args, return_value):
        return_value.append(function(args))

    ret_value = Manager().list()
    p = Process(target=return_handler_function, args=(function, args, ret_value))
    p.start()
    p.join(time)
    if p.is_alive():
        p.terminate()
        return None

    return list(ret_value)


# LOADING UTILS
# ------------------------------------------


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

