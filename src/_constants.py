"""
This file contains just a bunch of constants used in the generator. The value assigned to each constant is not significative, is just an integer used
to consume the lowest possible memory. Basically, this is just a sort of list of words coded as number for efficiency.
"""


# ACCOUNTS CONSTANTS
# ------------------------------------------

class ACCTS_DF_ATTR:
    S_ID = 'id'
    S_BUSINESS = 'business'
    S_BALANCE = 'balance'
    S_SCHED_BALANCE = 'scheduling_balance'
    S_NATIONALITY = 'nationality'
    S_BANK_ID = 'bank_id'
    S_AVG_TX_PER_STEP = 'avg_tx_per_step'
    S_COMPROM_RATIO = 'compromising_ratio'
    S_ROLE = 'role'
    S_FAN_IN_NAME = 'fan_in'
    S_FAN_OUT_NAME = 'fan_out'


class ACCTS_ROLES:
    NORMAL = -1
    ML = 0
    ML_MULE = 1
    ML_SOURCE = 2
    ML_LAYER = 3
    ML_DESTINATION = 4


class ACCTS_BUSINESS:
    NONE = 0
    RETAIL = 1
    CORPORATE = 2
    BUSINESS_TYPE = {'None': NONE, 'Retail': RETAIL, 'Corporate': CORPORATE}


# POPULATION CONSTANTS
# ------------------------------------------

class POPULATION:
    LAUNDERER_CREATION_SIMPLE_MODE = 0
    LAUNDERER_CREATION_STRUCTURED_MODE = 1


class COMM_TYPE:
    FULL_RANDOM = 0
    RANDOM = 1
    STRUCTURED_RANDOM = 2
    FROM_FILE = 3


# PATTERN CONSTANTS
# ------------------------------------------

class PTRN_TYPE:
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


class PTRN_ROLE:
    SOURCE = ACCTS_ROLES.ML_SOURCE
    LAYER = ACCTS_ROLES.ML_LAYER
    DESTINATION = ACCTS_ROLES.ML_DESTINATION


# SCHEDULING CONSTANTS
# ------------------------------------------

class SCHEDULING:
    # Scheduling
    INSTANT = 0
    PERIODIC = 1
    RANDOM = 2




