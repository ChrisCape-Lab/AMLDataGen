"""
This files contains a bunch of 'variables' that can be set through the config file. From the program point of view they are basically constants but
are initially set from the config file
"""
import src._constants as _c


class SIM:
    DEF_END_TIME = 54
    DEF_LAUNDERERS_CREATION_MODE = _c.POPULATION.LAUNDERER_CREATION_SIMPLE_MODE
    DEF_ALLOW_RANDOM_TXS = True
    DEF_SCHEDULE_CASHES_WITH_ML_PATTERNS = True


class ACCOUNT:
    # Default variance for the balance limits
    DEF_BALANCE_LIMIT_VARIANCE = 0.05

    # Default variance for the avg transactions per time
    DEF_AVG_TX_PER_TIME_VARIANCE = 0.5

    # Default values for cash in/out transactions
    DEF_CASH_IN_PROB = 0.75
    DEF_CASH_OUT_PROB = 0.85
    DEF_CASH_TX_MIN = 4
    DEF_CASH_TX_MAX = 8

    # Default values for the new beneficiary ratios
    __DEF_NEW_BENE_RATIO = 0.1
    __DEF_RETAIL_NEW_BENE_RATIO = 0.1
    __DEF_CORPORATE_NEW_BENE_RATIO = 0.05
    NEW_BENE_RATIOS = {None: __DEF_NEW_BENE_RATIO, _c.ACCTS_BUSINESS.RETAIL: __DEF_RETAIL_NEW_BENE_RATIO,
                       _c.ACCTS_BUSINESS.CORPORATE: __DEF_CORPORATE_NEW_BENE_RATIO}

    # Default values for the new neighbour ratios
    __DEF_NEW_NEIGHBOUR_RATIO = 0.5
    __DEF_RETAIL_NEW_NEIGHBOUR_RATIO = 0.5
    __DEF_CORPORATE_NEW_NEIGHBOUR_RATIO = 0.4
    NEW_NEIGHBOUR_RATIOS = {None: __DEF_NEW_NEIGHBOUR_RATIO, _c.ACCTS_BUSINESS.RETAIL: __DEF_RETAIL_NEW_NEIGHBOUR_RATIO,
                            _c.ACCTS_BUSINESS.CORPORATE: __DEF_CORPORATE_NEW_NEIGHBOUR_RATIO}


class POPULATION:
    # Ratio between launderers and total accounts
    DEF_ML_LAUNDERERS_RATIO = 0.05

    # Percentages that represent the distribution of money laundering sources, layerer and destinations
    DEF_ML_SOURCES_PERCENTAGE = 0.25
    DEF_ML_LAYERER_PERCENTAGE = 0.5
    DEF_ML_DESTINATIONS_PERCENTAGE = 0.25

    assert DEF_ML_SOURCES_PERCENTAGE + DEF_ML_LAYERER_PERCENTAGE + DEF_ML_DESTINATIONS_PERCENTAGE == 1

    DEF_ML_LIMITING_QUANTILE = 0.9


class COMM:
    DEF_RND_COMMUNITY_DENSITY = 0.3

    DEF_MIN_KNOWN_NODES = 5
    DEF_MAX_KNOWN_NODES = 20

    DEF_MIN_COMM_SIZE = 30
    DEF_MAX_COMM_SIZE = 100

    DEF_COMMUNITY_TYPE = _c.COMM_TYPE.FULL_RANDOM


class PATTERN:
    REPEATED_MIN = 4
    REPEATED_MAX = 8
    BIPARTITE_MIN_SOURCES = 0.2
    BIPARTITE_MAX_SOURCES = 0.3
    BIPARTITE_MIN_DESTINATIONS = 0.2
    BIPARTITE_MAX_DESTINATIONS = 0.3
    BIPARTITE_MIN_LAYERS_NUM = 2
    BIPARTITE_MAX_LAYERS_NUM = 6
    BIPARTITE_EDGE_DENSITY = 0.8
    CASH_NUM_MIN = 3
    CASH_NUM_MAX = 5


def load_variables(variables: dict) -> None:

    return
