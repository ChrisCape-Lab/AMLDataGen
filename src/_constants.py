"""
This file contains just a bunch of constants used in the generator. The value assigned to each constant is not significative, is just an integer used
to consume the lowest possible memory. Basically, this is just a sort of list of words coded as number for efficiency.
"""
from src._variables import *


class GENERAL:
    BALANCE = 'balance'
    FAN_IN = 'fan_in'
    FAN_OUT = 'fan_out'



class ACCOUNT:
    # Accounts Classes
    ML = -1
    NORMAL = 0
    ML_MULE = 1
    ML_SOURCE = 2
    ML_LAYER = 3
    ML_DESTINATION = 4

    # Accounts constants dictionaries
    NONE = 0
    RETAIL = 1
    CORPORATE = 2
    BUSINESS_TYPE = {'None': NONE, 'Retail': RETAIL, 'Corporate': CORPORATE}

    # Accounts new beneficiary and neighbours array ratios
    NEW_BENE_RATIOS = {NONE: DEFAULT_NEW_BENE_RATIO, RETAIL: RETAIL_NEW_BENE_RATIO, CORPORATE: CORPORATE_NEW_BENE_RATIO}
    NEW_NEIGHBOUR_RATIOS = {NONE: DEFAULT_NEW_NEIGHBOUR_RATIO, RETAIL: RETAIL_NEW_NEIGHBOUR_RATIO, CORPORATE: CORPORATE_NEW_NEIGHBOUR_RATIO}


class POPULATION:

    SIMPLE_MODE = 0
    STRUCTURED_MODE = 1


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





# AMOUNT RATIO CATEGORIES
NORMAL_AMT = 0
ROUNDED = 1
UNDER_THRESH = 2


# Transaction particular Patterns
FAN_IN = 0
FAN_OUT = 1
CYCLE = 2
SCATTER_GATHER = 3
GATHER_SCATTER = 4
U = 5
REPEATED = 6
BIPARTITE = 7
CASH_IN = 8
CASH_OUT = 9


# Scheduling
INSTANTS = 0
PERIODIC = 1
RANDOM = 2


# 'Hardcoded' values
REPEATED_MIN = 8
REPEATED_MAX = 12
CASH_TX_MIN = 4
CASH_TX_MAX = 8


if __name__ == "__main__":
    import networkx as nx

    G = nx.Graph()
    G.add_edge(1, 2, color='red', weight=0.84, size=300)

    for key, value in G[1].items():
        print(key, value)






