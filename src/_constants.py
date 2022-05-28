"""
This file contains just a bunch of constants used in the generator. The value assigned to each constant is not significative, is just an integer used
to consume the lowest possible memory. Basically, this is just a sort of list of words coded as number for efficiency.
"""
import pandas as pd


class GENERAL:
    AVAILABLE_BALANCE = 'available_balance'
    BALANCE = 'balance'
    FAN_IN_NAME = 'fan_in'
    FAN_OUT_NAME = 'fan_out'

    # Transaction particular Patterns
    RANDOM = -1
    FAN_IN = 0
    FAN_OUT = 1
    CYCLE = 2
    SCATTER_GATHER = 3
    GATHER_SCATTER = 4
    U = 5
    REPEATED = 6
    BIPARTITE = 7
    RANDOM_P = 8
    CASH_IN = 9
    CASH_OUT = 10

    TOT_PATTERNS = 8


class ACCOUNT:
    # Accounts Classes
    NORMAL = -1
    ML = 0
    ML_MULE = 1
    ML_SOURCE = 2
    ML_LAYER = 3
    ML_DESTINATION = 4

    # Accounts constants dictionaries
    NONE = 0
    RETAIL = 1
    CORPORATE = 2
    BUSINESS_TYPE = {'None': NONE, 'Retail': RETAIL, 'Corporate': CORPORATE}


class POPULATION:
    LAUNDERER_CREATION_SIMPLE_MODE = 0
    LAUNDERER_CREATION_STRUCTURED_MODE = 1


# Community Constants
class COMMUNITY:
    FULL_RANDOM = 0
    RANDOM = 1
    STRUCTURED_RANDOM = 2
    FROM_FILE = 3


class PATTERN:
    SOURCE = 0
    LAYER = 1
    DESTINATION = 2


class SCHEDULING:
    # Scheduling
    INSTANT = 0
    PERIODIC = 1
    RANDOM = 2


if __name__ == "__main__":
    import networkx as nx

    graph = nx.Graph()
    graph.add_node(0, {'attr1': 1, 'attr2': 2})
    graph.add_node(1)

    print(graph[0])






