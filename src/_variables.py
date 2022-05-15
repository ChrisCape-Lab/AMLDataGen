"""
This files contains a bunch of 'variables' that can be set through the config file. From the program point of view they are basically constants but
are initially set from the config file
"""
import src._constants as c


class SIMULATION:
    END_TIME = 54
    LAUNDERERS_CREATION_MODE = c.POPULATION.LAUNDERER_CREATION_SIMPLE_MODE


class ACCOUNT:
    DEF_BALANCE_LIMIT_VARIANCE = 0.05
    DEF_AVG_TX_PER_TIME_VARIANCE = 0.5
    DEF_CASH_IN_PROB = 0.75
    DEF_CASH_OUT_PROB = 0.85

    CASH_TX_MIN = 4
    CASH_TX_MAX = 8

    DEF_NEW_BENE_RATIO = 0.1
    RETAIL_NEW_BENE_RATIO = 0.1
    CORPORATE_NEW_BENE_RATIO = 0.05

    DEFAULT_NEW_NEIGHBOUR_RATIO = 0.5
    RETAIL_NEW_NEIGHBOUR_RATIO = 0.5
    CORPORATE_NEW_NEIGHBOUR_RATIO = 0.4

    # Accounts new beneficiary and neighbours array ratios
    NEW_BENE_RATIOS = {None: DEF_NEW_BENE_RATIO, c.ACCOUNT.RETAIL: RETAIL_NEW_BENE_RATIO,
                       c.ACCOUNT.CORPORATE: CORPORATE_NEW_BENE_RATIO}
    NEW_NEIGHBOUR_RATIOS = {None: DEFAULT_NEW_NEIGHBOUR_RATIO, c.ACCOUNT.RETAIL: RETAIL_NEW_NEIGHBOUR_RATIO,
                            c.ACCOUNT.CORPORATE: CORPORATE_NEW_NEIGHBOUR_RATIO}


class POPULATION:
    # Ratio between launderers and total accounts
    DEF_ML_LAUNDERERS_RATIO = 0.05

    # Percentages that represent the distribution of money laundering sources, layerer and destinations
    DEF_ML_SOURCES_PERCENTAGE = 0.25
    DEF_ML_LAYERER_PERCENTAGE = 0.5
    DEF_ML_DESTINATIONS_PERCENTAGE = 0.25

    assert DEF_ML_SOURCES_PERCENTAGE + DEF_ML_LAYERER_PERCENTAGE + DEF_ML_DESTINATIONS_PERCENTAGE == 1

    DEF_ML_LIMITING_QUANTILE = 0.9


class COMMUNITY:
    DEF_RND_COMMUNITY_DENSITY = 0.3

    DEF_MIN_KNOWN_NODES = 5
    DEF_MAX_KNOWN_NODES = 20

    DEF_MIN_COMM_SIZE = 30
    DEF_MAX_COMM_SIZE = 100


class PATTERN:
    BIPARTITE_MIN_SOURCES = 0.2
    BIPARTITE_MAX_SOURCES = 0.3
    BIPARTITE_MIN_DESTINATIONS = 0.2
    BIPARTITE_MAX_DESTINATIONS = 0.3
    BIPARTITE_MIN_LAYERS_NUM = 2
    BIPARTITE_MAX_LAYERS_NUM = 6
    BIPARTITE_EDGE_DENSITY = 0.8


def load_variables(variables: dict) -> None:

    return
