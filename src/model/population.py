import random
import numpy as np
import pandas as pd

from src._constants import ACCOUNT, POPULATION
from src._variables import ACCOUNT, POPULATION
from src.utils import add_to_dict_of_list


class Bank:
    def __init__(self, bank_id, compromising_ratio, launderer_ratio):
        self.id = bank_id
        self.compromising_ratio = compromising_ratio
        self.launderer_ratio = launderer_ratio


class Account:
    def __init__(self, acct_id, balance, balance_limit_percentage, business, behaviours, bank_id, avg_tx_per_step, min_amount, max_amount,
                 compromising_ratio, role):
        # Unique ID that define the account
        self.id = acct_id

        # Type of business of the account, one among RETAIL or CORPORATE
        self.business = ACCOUNT.BUSINESS_TYPE[business]

        #
        self.balance = balance
        self.balance_limits = self.__set_balance_limits(balance_limit_percentage)

        self.available_balance = self.balance

        # The available behaviours for patterns of the account
        self.behaviours = behaviours

        # Nationality of the account owner [For now is unused, may be used for Money-Launderers creation or compromising ratio]
        self.nationality = None
        # The ID of user account's bank
        self.bank_id = bank_id

        #
        self.avg_tx_per_step = avg_tx_per_step
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.new_beneficiary_ratio = random.gauss(ACCOUNT.NEW_BENE_RATIOS[self.business], 0.002)
        self.new_neighbour_ratio = random.gauss(ACCOUNT.NEW_NEIGHBOUR_RATIOS[self.business], 0.05)

        self.compromising_ratio = compromising_ratio
        self.role = role

    # CLASS METHODS
    # ------------------------------------------

    @classmethod
    def get_dataframe_columns(cls):
        return ['id', 'business', 'balance', 'avail_balance', 'nationality', 'bank_id', 'avg_tx_per_step', 'compromising_ratio', 'role']

    # PRIVATE INITIALIZERS METHODS
    # ------------------------------------------

    def __set_balance_limits(self, balance_limit_percentage):
        balance_limit_min = random.randrange(balance_limit_percentage - ACCOUNT.DEF_BALANCE_LIMIT_VARIANCE,
                                             balance_limit_percentage + ACCOUNT.DEF_BALANCE_LIMIT_VARIANCE)
        balance_limit_max = random.randrange(balance_limit_percentage - ACCOUNT.DEF_BALANCE_LIMIT_VARIANCE,
                                             balance_limit_percentage + ACCOUNT.DEF_BALANCE_LIMIT_VARIANCE)

        return [self.balance * (1-balance_limit_min), self.balance * (1+balance_limit_max)]

    # METHODS
    # ------------------------------------------

    def has_amount(self, amount):
        return self.balance > amount

    def require_cash_in(self):
        return self.balance < min(self.balance_limits)

    def require_cash_out(self):
        return self.balance > max(self.balance_limits)

    def update_balance(self, amount):
        self.balance = self.balance + amount

    def to_dataframe_row(self):
        return [self.id, self.business, self.balance, self.available_balance, self.nationality, self.bank_id, self.avg_tx_per_step,
                self.compromising_ratio, self.role]


class Population:
    def __init__(self):
        self.banks = dict()
        self.accounts = dict()

        self.bank_to_acc = dict()
        self.role_to_acc = dict()
        self.patterns_to_acc = dict()
        self.compromising_accounts_ratio = list()
        self.compromised = set()

    # GETTERS
    # ------------------------------------------
    def get_bank_ids(self) -> list:
        return list(self.accounts.keys())

    def get_bank_nums(self) -> int:
        return len(self.accounts)

    def get_accounts_ids(self) -> list:
        return list(self.accounts.keys())

    def get_accounts_num(self) -> int:
        return len(self.accounts)

    def get_accounts_from_bank(self, bank_id: int) -> list:
        return self.bank_to_acc[bank_id]

    def get_accounts_from_pattern(self, pattern: int) -> list:
        return self.patterns_to_acc[pattern]

    def get_accounts_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([acc.to_dataframe_row for _, acc in self.accounts.items()], columns=Account.get_dataframe_columns())

    # MODIFIERS
    # ------------------------------------------
    def add_bank(self, bank: Bank) -> None:
        self.banks[bank.id] = bank

    def add_account(self, account: Account) -> None:
        assert account.bank_id in self.banks.keys()

        # Add account to general dictionary
        account_id = account.id
        self.accounts[account_id] = account

        # Add account to Bank->Account.id dictionary
        bank_id = account.bank_id
        add_to_dict_of_list(self.bank_to_acc, bank_id, account_id)

        # Add account to Pattern->Account.id dictionary
        patterns = account.behaviours
        for pattern in patterns:
            add_to_dict_of_list(self.patterns_to_acc, pattern, account_id)

        self.compromising_accounts_ratio.append(account.compromising_ratio)

    def add_compromised(self, account_id: int) -> None:
        self.compromised.add(account_id)

    # CREATION
    # ------------------------------------------

    def create_launderers(self, mode, limiting_quantile=POPULATION.DEF_ML_LIMITING_QUANTILE):
        if mode == POPULATION.SIMPLE_MODE:
            self.__create_simple_launderers(limiting_quantile)
        else:
            self.__create_structured_launderers(limiting_quantile)

    def __create_simple_launderers(self, limiting_quantile: float) -> int:
        num_accounts = len(self.accounts)
        num_banks = len(self.bank_to_acc)
        launderers_num = int(num_accounts * POPULATION.DEF_ML_LAUNDERERS_RATIO)

        # Create dataframe from accounts. It is done to speed up the computation
        acc_df = self.get_accounts_dataframe()

        # The following operations are used to remove the most active accounts. This is done by first extracting the 0.9 quantile from the avg_fan_out
        # distribution adn then removing the account with higher avg_fan_out. Launderers does not do many transactions in order to not be bothered
        if limiting_quantile > 0.0:
            quantile_df = acc_df[['avg_tx_per_step']]
            tx_out_quant = np.quantile(quantile_df, limiting_quantile, axis=0)
            acc_df.drop(acc_df[acc_df['avg_tx_per_step'] >= tx_out_quant].index, inplace=True)

        launderers_ids = random.sample(self.accounts.keys(), k=launderers_num)
        for i in launderers_ids:
            self.accounts[i].role = ACCOUNT.ML
            add_to_dict_of_list(self.role_to_acc, ACCOUNT.ML_SOURCE, i)
            add_to_dict_of_list(self.role_to_acc, ACCOUNT.ML_LAYER, i)
            add_to_dict_of_list(self.role_to_acc, ACCOUNT.ML_DESTINATION, i)

        num_launderers = sum([len(l) for _, l in self.role_to_acc.items()])
        assert num_launderers in range(int(launderers_num * 0.9), int(launderers_num * 1.1))

        return num_launderers

    def __create_structured_launderers(self, limiting_quantile: float) -> int:
        num_accounts = len(self.accounts)
        num_banks = len(self.bank_to_acc)
        launderers_num = int(num_accounts * POPULATION.DEF_ML_LAUNDERERS_RATIO)
        banks_laundering_ratio = [bank.launderer_ratio for bank in self.banks]

        # Each one is a list which contains, for each bank, the number of source/layer/dest nodes related to that bank
        num_sources = launderers_num * POPULATION.DEF_ML_SOURCES_PERCENTAGE
        source_dist = banks_laundering_ratio * num_sources

        num_layerer = launderers_num * POPULATION.DEF_ML_LAYERER_PERCENTAGE
        layerer_dist = banks_laundering_ratio * num_layerer

        num_destinations = launderers_num * POPULATION.DEF_ML_DESTINATIONS_PERCENTAGE
        destination_dist = banks_laundering_ratio * num_destinations

        # Create dataframe from accounts. It is done to speed up the computation
        acc_df = self.get_accounts_dataframe()

        # The following operations are used to remove the most active accounts. This is done by first extracting the 0.9 quantile from the avg_fan_out
        # distribution adn then removing the account with higher avg_fan_out. Launderers does not do many transactions in order to not be bothered
        if limiting_quantile > 0.0:
            quantile_df = acc_df[['avg_tx_per_step']]
            tx_out_quant = np.quantile(quantile_df, limiting_quantile, axis=0)
            acc_df.drop(acc_df[acc_df['avg_tx_per_step'] >= tx_out_quant].index, inplace=True)

        # For each bank are extracted sources, layers and destinations
        for bank_id in range(0, num_banks):
            bank_users_idxs = set(acc_df[acc_df[[5] == bank_id]][[0]])

            # Set AMLSources for bank_i
            source_ids = random.sample(bank_users_idxs, k=source_dist[bank_id])
            for i in source_ids:
                self.accounts[i].role = ACCOUNT.ML_SOURCE
                add_to_dict_of_list(self.role_to_acc, ACCOUNT.ML_SOURCE, i)

            # Set AMLLayer for bank_i
            bank_users_idxs = bank_users_idxs - set(source_ids)
            layer_ids = random.sample(bank_users_idxs, k=layerer_dist[bank_id])
            for i in layer_ids:
                self.accounts[i].role = ACCOUNT.ML_LAYER
                add_to_dict_of_list(self.role_to_acc, ACCOUNT.ML_LAYER, i)

            # Set AMLDestination for bank_i
            bank_users_idxs = bank_users_idxs - set(layer_ids)
            destination_ids = random.sample(bank_users_idxs, k=destination_dist[bank_id])
            for i in destination_ids:
                self.accounts[i].role = ACCOUNT.ML_DESTINATION
                add_to_dict_of_list(self.role_to_acc, ACCOUNT.ML_DESTINATION, i)

        num_launderers = sum([len(l) for _, l in self.role_to_acc.items()])
        assert num_launderers in range(int(launderers_num * 0.9), int(launderers_num * 1.1))

        return num_launderers

    # BEHAVIOUR
    # ------------------------------------------

    def send_transaction(self, originator_id: int, beneficiary_id: int, amount: int) -> bool:
        originator = self.accounts[originator_id]
        beneficiary = self.accounts[beneficiary_id]

        outcome = originator.has_amount()
        if outcome:
            originator.update_balance(-amount)
            beneficiary.update_balance(amount)

        return outcome

    def get_compromised(self, num_compromised: int) -> set:
        # Get compromised accounts based on compromised probability
        compromised = np.random.choice(len(self.accounts), size=num_compromised, replace=False, p=self.compromising_accounts_ratio)
        # Check whether some chosen account has already been compromised in the past...
        already_compromised = [acc in compromised for acc in self.compromised]
        # ...if so add those accounts to a 'to remove' list...
        to_remove = already_compromised

        # Keep choosing accounts until only available accounts are found
        while len(already_compromised) > 0:
            compromised = np.random.choice(len(self.accounts), size=num_compromised, replace=False, p=self.compromising_accounts_ratio)
            already_compromised = [acc in compromised for acc in self.compromised]
            to_remove.extend(already_compromised)

        # ...and then remove them. They have used the opportunity to be not compromised a second time
        self.compromised = self.compromised - set(to_remove)

        self.compromised = set.union(self.compromised, set(compromised))

        return compromised











