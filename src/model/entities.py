import random

import src._constants as _c
import src._variables as _v


class Bank:
    def __init__(self, bank_id, name, compromising_ratio, launderer_ratio):
        self.id = bank_id
        self.name = name
        self.compromising_ratio = compromising_ratio
        self.launderer_ratio = launderer_ratio


class Account:
    def __init__(self, acct_id: int, balance: float, balance_limit_percentage: float, business: str, behaviours: str,
                 bank_id: int, avg_tx_per_step: float,
                 min_amount: float, max_amount: float, compromising_ratio: float, role: int):
        assert balance >= 0, "Balance must be greater than zero, found " + str(balance)
        assert avg_tx_per_step >= 0, "avg_tx_per_step must be greater than zero, found " + str(avg_tx_per_step)
        assert min_amount >= 0, "min_amount must be greater than zero, found " + str(min_amount)
        assert max_amount >= 0, "max_amount must be greater than zero, found " + str(max_amount)
        # assert compromising_ratio >= 0, "compromising_ratio must be greater than zero, found " + str(compromising_ratio)

        # Unique ID that define the account
        self.id = int(acct_id)

        # Type of business of the account, one among RETAIL or CORPORATE
        self.business = _c.ACCTS_BUSINESS.BUSINESS_TYPE[business]

        #
        self.balance = round(balance, 2)

        # Balance limit is the balance that triggers a cash in or cash-out and corresponds to balance% and balance(2-%)
        self.balance_limits = self.__set_balance_limits(balance_limit_percentage)

        self.available_balance = round(balance, 2)

        # The available behaviours for patterns of the account
        self.behaviours = self.__handle_behaviours(behaviours)

        # Nationality of the account owner [For now is unused, may be used for Money-Launderers creation or compromising ratio]
        self.nationality = None
        # The ID of user account's bank
        self.bank_id = bank_id

        #
        self.avg_tx_per_step = avg_tx_per_step
        self.min_amount = round(min_amount, 2)
        self.max_amount = round(max_amount, 2)
        self.new_beneficiary_ratio = random.gauss(_v.ACCOUNT.NEW_BENE_RATIOS[self.business], 0.002)
        self.new_neighbour_ratio = random.gauss(_v.ACCOUNT.NEW_NEIGHBOUR_RATIOS[self.business], 0.05)

        self.compromising_ratio = compromising_ratio
        self.role = role

    # CLASS METHODS
    # ------------------------------------------

    @classmethod
    def get_dataframe_columns(cls) -> list:
        return [_c.ACCTS_DF_ATTR.S_ID, _c.ACCTS_DF_ATTR.S_BUSINESS, _c.ACCTS_DF_ATTR.S_BALANCE, _c.ACCTS_DF_ATTR.S_SCHED_BALANCE,
                _c.ACCTS_DF_ATTR.S_NATIONALITY, _c.ACCTS_DF_ATTR.S_BANK_ID, _c.ACCTS_DF_ATTR.S_AVG_TX_PER_STEP, _c.ACCTS_DF_ATTR.S_COMPROM_RATIO,
                _c.ACCTS_DF_ATTR.S_ROLE]

    @classmethod
    def get_dataframe_column_type(cls) -> dict:
        return {_c.ACCTS_DF_ATTR.S_ID: int, _c.ACCTS_DF_ATTR.S_BUSINESS: int, _c.ACCTS_DF_ATTR.S_BALANCE: float,
                _c.ACCTS_DF_ATTR.S_SCHED_BALANCE: float, _c.ACCTS_DF_ATTR.S_NATIONALITY: int, _c.ACCTS_DF_ATTR.S_BANK_ID: int,
                _c.ACCTS_DF_ATTR.S_AVG_TX_PER_STEP: float, _c.ACCTS_DF_ATTR.S_COMPROM_RATIO: float, _c.ACCTS_DF_ATTR.S_ROLE: int}

    # PRIVATE INITIALIZERS METHODS
    # ------------------------------------------

    @staticmethod
    def __handle_behaviours(behaviours_in: str) -> list:
        if behaviours_in == 'all':
            return [i for i in range(0, _c.PTRN_TYPE.TOT_PATTERNS)]

        behaviours_list = behaviours_in.split("_")
        behaviours_list = [int(x) for x in behaviours_list]

        return behaviours_list

    def __set_balance_limits(self, balance_limit_percentage: float) -> list:
        balance_limit_min = random.uniform(balance_limit_percentage - _v.ACCOUNT.DEF_BALANCE_LIMIT_VARIANCE,
                                           balance_limit_percentage + _v.ACCOUNT.DEF_BALANCE_LIMIT_VARIANCE)
        balance_limit_max = random.uniform(balance_limit_percentage - _v.ACCOUNT.DEF_BALANCE_LIMIT_VARIANCE,
                                           balance_limit_percentage + _v.ACCOUNT.DEF_BALANCE_LIMIT_VARIANCE)

        return [self.balance * (1 - balance_limit_min), self.balance * (2 - balance_limit_max)]

    # GETTERS
    # ------------------------------------------
    @staticmethod
    def get_number_of_cash_tx() -> int:
        return random.randint(_v.ACCOUNT.DEF_CASH_TX_MIN, _v.ACCOUNT.DEF_CASH_TX_MAX)

    def get_cash_in_amount(self) -> float:
        return round(random.uniform(self.min_amount, self.max_amount), 2)

    def get_cash_out_amount(self) -> float:
        if self.max_amount > self.balance:
            return 0

        return round(random.uniform(self.min_amount, self.max_amount), 2)

    def get_number_of_txs(self) -> int:
        return int(random.gauss(self.avg_tx_per_step, _v.ACCOUNT.DEF_AVG_TX_PER_TIME_VARIANCE))

    def get_tx_amount(self) -> float:
        min_amount = min(self.min_amount, self.balance)
        max_amount = min(self.max_amount, self.balance)

        return round(random.uniform(min_amount, max_amount), 2)

    def to_dataframe_row(self) -> list:
        return [int(self.id), int(self.business), float(self.balance), float(self.available_balance), self.nationality,
                int(self.bank_id), float(self.avg_tx_per_step), float(self.compromising_ratio), self.role]

    # CHECKERS
    # ------------------------------------------

    def has_amount(self, amount):
        assert amount > 0
        return (self.balance > amount) and (self.available_balance > amount)

    def require_cash_in(self):
        return (self.balance < min(self.balance_limits)) or (self.available_balance < min(self.balance_limits))

    def require_cash_out(self):
        return self.balance > max(self.balance_limits)

    # MODIFIERS
    # ------------------------------------------
    def update_available_balance(self, amount):
        assert self.available_balance + amount > 0, "Not enough balance: requested " + str(
            -amount) + " but available " + str(self.available_balance)
        self.available_balance += amount

    def update_balance(self, amount):
        assert self.balance + amount > 0, "Not enough balance: requested " + str(-amount) + " but available " + str(
            self.balance)
        self.balance += amount
