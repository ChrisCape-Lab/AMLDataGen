"""
This file contains just a bunch of constants used in the generator. The value assigned to each constant is not significative, is just an integer used
to consume the lowest possible memory. Basically, this is just a sort of list of words coded as number for efficiency.
"""

# Accounts Classes
NORMAL = 0
AML_MULE = 1
AML_SOURCE = 2
AML_LAYER = 3
AML_DESTINATION = 4


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


DEFAULT_NEW_BENE_RATIO = {0: 0.1, 1: 0.05}
DEFAULT_NEW_NEIGHBOUR_RATIO = {0: 0.5, 1: 0.4}



