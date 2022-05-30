"""
This files contains a bunch of 'variables' that can be set through the config file. From the program point of view they are basically constants but
are initially set from the config file
"""
import src._constants as _c


class SIMULATION:
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
    DEF_RND_CASH_TXS_MIN = 2
    DEF_RND_CASH_TXS_MAX = 4

    # Default values for the new beneficiary ratios
    __DEF_NEW_BENE_RATIO = 0.1
    __DEF_RETAIL_NEW_BENE_RATIO = 0.1
    __DEF_CORPORATE_NEW_BENE_RATIO = 0.05
    NEW_BENE_RATIOS = {None: __DEF_NEW_BENE_RATIO, _c.ACCOUNT.RETAIL: __DEF_RETAIL_NEW_BENE_RATIO,
                       _c.ACCOUNT.CORPORATE: __DEF_CORPORATE_NEW_BENE_RATIO}

    # Default values for the new neighbour ratios
    __DEF_NEW_NEIGHBOUR_RATIO = 0.5
    __DEF_RETAIL_NEW_NEIGHBOUR_RATIO = 0.5
    __DEF_CORPORATE_NEW_NEIGHBOUR_RATIO = 0.4
    NEW_NEIGHBOUR_RATIOS = {None: __DEF_NEW_NEIGHBOUR_RATIO, _c.ACCOUNT.RETAIL: __DEF_RETAIL_NEW_NEIGHBOUR_RATIO,
                            _c.ACCOUNT.CORPORATE: __DEF_CORPORATE_NEW_NEIGHBOUR_RATIO}


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

    DEF_COMMUNITY_TYPE = _c.COMMUNITY.FULL_RANDOM


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


def __launderer_creation_to_constant(mode: str) -> int:
    if mode == 'simple':
        return _c.POPULATION.LAUNDERER_CREATION_SIMPLE_MODE
    elif mode == 'structured':
        return _c.POPULATION.LAUNDERER_CREATION_STRUCTURED_MODE
    else:
        raise NotImplementedError


def __community_to_constant(community: str) -> int:
    if community == 'full_random':
        return _c.COMMUNITY.FULL_RANDOM
    elif community == 'random':
        return _c.COMMUNITY.RANDOM
    elif community == 'structured':
        return _c.COMMUNITY.STRUCTURED_RANDOM
    elif community == 'from_file':
        return _c.COMMUNITY.FROM_FILE
    else:
        raise NotImplementedError


def load_variables(variables: dict) -> None:
    # Simulation variables
    SIMULATION.DEF_END_TIME = variables['end_time']
    SIMULATION.DEF_LAUNDERERS_CREATION_MODE = __launderer_creation_to_constant(variables['launderers_creation_mode'])
    SIMULATION.DEF_ALLOW_RANDOM_TXS = variables['allow_random_txs']
    SIMULATION.DEF_SCHEDULE_CASHES_WITH_ML_PATTERNS = variables['schedule_cashes_with_ML_patterns']

    # Account variables
    ACCOUNT.DEF_BALANCE_LIMIT_VARIANCE = variables['balance_limit_variance']
    ACCOUNT.DEF_AVG_TX_PER_TIME_VARIANCE = variables['avg_tx_per_time_variance']
    ACCOUNT.DEF_CASH_IN_PROB = variables['cash_in_probability']
    ACCOUNT.DEF_CASH_OUT_PROB = variables['cash_out_probability']
    ACCOUNT.DEF_RND_CASH_TXS_MIN = variables['random_cash_txs_min']
    ACCOUNT.DEF_RND_CASH_TXS_MAX = variables['random_cash_txs_max']
    ACCOUNT.__DEF_NEW_BENE_RATIO = variables['default_new_beneficiary_ratio']
    ACCOUNT.__DEF_RETAIL_NEW_BENE_RATIO = variables['retail_new_beneficiary_ratio']
    ACCOUNT.__DEF_CORPORATE_NEW_BENE_RATIO = variables['corporate_new_beneficiary_ratio']
    ACCOUNT.__DEF_NEW_NEIGHBOUR_RATIO = variables['default_new_neighbour_ratio']
    ACCOUNT.__DEF_RETAIL_NEW_NEIGHBOUR_RATIO = variables['retail_new_neighbour_ratio']
    ACCOUNT.__DEF_CORPORATE_NEW_NEIGHBOUR_RATIO = variables['corporate_new_neighbour_ratio']

    # Population variables
    POPULATION.DEF_ML_LAUNDERERS_RATIO = variables['ML_launderers_ratio']
    POPULATION.DEF_ML_SOURCES_PERCENTAGE = variables['ML_sources_percentage']
    POPULATION.DEF_ML_LAYERER_PERCENTAGE = variables['ML_layerer_percentage']
    POPULATION.DEF_ML_DESTINATIONS_PERCENTAGE = variables['ML_destinations_percentage']

    assert POPULATION.DEF_ML_SOURCES_PERCENTAGE + POPULATION.DEF_ML_LAYERER_PERCENTAGE +\
           POPULATION.DEF_ML_DESTINATIONS_PERCENTAGE == 1

    POPULATION.DEF_ML_LIMITING_QUANTILE = variables['ML_limiting_quantile']

    # Community variables
    COMMUNITY.DEF_RND_COMMUNITY_DENSITY = variables['random_community_density']
    COMMUNITY.DEF_MIN_KNOWN_NODES = variables['min_known_nodes']
    COMMUNITY.DEF_MAX_KNOWN_NODES = variables['max_known_nodes']
    COMMUNITY.DEF_MIN_COMM_SIZE = variables['min_community_size']
    COMMUNITY.DEF_MAX_COMM_SIZE = variables['max_community_size']
    COMMUNITY.DEF_COMMUNITY_TYPE = __community_to_constant(variables['community_type'])

    # Pattern variables
    PATTERN.REPEATED_MIN = variables['repeated_pttrn_min_requests']
    PATTERN.REPEATED_MAX = variables['repeated_pttrn_max_requests']
    PATTERN.BIPARTITE_MIN_SOURCES = variables['bipartite_pttrn_min_sources_percentage']
    PATTERN.BIPARTITE_MAX_SOURCES = variables['bipartite_pttrn_min_sources_percentage']
    PATTERN.BIPARTITE_MIN_DESTINATIONS = variables['bipartite_pttrn_min_dest_percentage']
    PATTERN.BIPARTITE_MAX_DESTINATIONS = variables['bipartite_pttrn_max_dest_percentage']
    PATTERN.BIPARTITE_MIN_LAYERS_NUM = variables['bipartite_min_layers_num']
    PATTERN.BIPARTITE_MAX_LAYERS_NUM = variables['bipartite_max_layers_num']
    PATTERN.BIPARTITE_EDGE_DENSITY = variables['bipartite_edge_density']
    PATTERN.CASH_NUM_MIN = variables['cash_num_min']
    PATTERN.CASH_NUM_MAX = variables['cash_num_max']

    return
