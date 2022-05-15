import random
import numpy as np
import pandas as pd

import src._constants as _c
import src._variables as _v
from src.utils import add_to_dict_of_list


class Bank:
    def __init__(self, bank_id, name, compromising_ratio, launderer_ratio):
        self.id = bank_id
        self.name = name
        self.compromising_ratio = compromising_ratio
        self.launderer_ratio = launderer_ratio


class Account:
    def __init__(self, acct_id, balance, balance_limit_percentage, business, behaviours, bank_id, avg_tx_per_step,
                 min_amount, max_amount, compromising_ratio, role):
        # Unique ID that define the account
        self.id = acct_id

        # Type of business of the account, one among RETAIL or CORPORATE
        self.business = _c.ACCOUNT.BUSINESS_TYPE[business]

        #
        self.balance = balance

        # Balance limit is the balance that triggers a cash in or cash-out and corresponds to balance% and balance(2-%)
        self.balance_limits = self.__set_balance_limits(balance_limit_percentage)

        self.available_balance = self.balance

        # The available behaviours for patterns of the account
        self.behaviours = self.__handle_behaviours(behaviours)

        # Nationality of the account owner [For now is unused, may be used for Money-Launderers creation or compromising ratio]
        self.nationality = None
        # The ID of user account's bank
        self.bank_id = bank_id

        #
        self.avg_tx_per_step = avg_tx_per_step
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.new_beneficiary_ratio = random.gauss(_v.ACCOUNT.NEW_BENE_RATIOS[self.business], 0.002)
        self.new_neighbour_ratio = random.gauss(_v.ACCOUNT.NEW_NEIGHBOUR_RATIOS[self.business], 0.05)

        self.compromising_ratio = compromising_ratio
        self.role = role

    # CLASS METHODS
    # ------------------------------------------

    @classmethod
    def get_dataframe_columns(cls):
        return ['id', 'business', 'balance', 'avail_balance', 'nationality', 'bank_id', 'avg_tx_per_step',
                'compromising_ratio', 'role']

    # PRIVATE INITIALIZERS METHODS
    # ------------------------------------------

    def __handle_behaviours(self, behaviours_in: str) -> list:
        if behaviours_in == 'all':
            return [i for i in range(0, _c.GENERAL.TOT_PATTERNS)]

        behaviours_in = behaviours_in[1:-2]
        behaviours_list = behaviours_in.split("_")
        behaviours_list = [int(x) for x in behaviours_list]

        return behaviours_list

    def __set_balance_limits(self, balance_limit_percentage):
        balance_limit_min = random.uniform(balance_limit_percentage - _v.ACCOUNT.DEF_BALANCE_LIMIT_VARIANCE,
                                             balance_limit_percentage + _v.ACCOUNT.DEF_BALANCE_LIMIT_VARIANCE)
        balance_limit_max = random.uniform(balance_limit_percentage - _v.ACCOUNT.DEF_BALANCE_LIMIT_VARIANCE,
                                             balance_limit_percentage + _v.ACCOUNT.DEF_BALANCE_LIMIT_VARIANCE)

        return [self.balance * (1 - balance_limit_min), self.balance * (2 - balance_limit_max)]

    # GETTERS
    # ------------------------------------------
    def get_number_of_cash_tx(self):
        return random.randint(_v.ACCOUNT.CASH_TX_MIN, _v.ACCOUNT.CASH_TX_MAX)

    def get_cash_amount(self):
        return random.uniform(self.min_amount, self.max_amount)

    def get_number_of_txs(self):
        return int(random.gauss(self.avg_tx_per_step, _v.ACCOUNT.DEF_AVG_TX_PER_TIME_VARIANCE))

    def get_tx_amount(self):
        min_amount = min(self.min_amount, self.balance)
        max_amount = min(self.max_amount, self.balance)

        return round(random.uniform(min_amount, max_amount), 2)

    def to_dataframe_row(self):
        return [self.id, self.business, self.balance, self.available_balance, self.nationality, self.bank_id,
                self.avg_tx_per_step,
                self.compromising_ratio, self.role]

    # CHECKERS
    # ------------------------------------------

    def has_amount(self, amount):
        return self.balance > amount

    def require_cash_in(self):
        return self.balance < min(self.balance_limits) or self.available_balance < min(self.balance_limits)

    def require_cash_out(self):
        return self.balance > max(self.balance_limits)

    # MODIFIERS
    # ------------------------------------------
    def update_available_balance(self, amount):
        self.available_balance -= amount

    def update_balance(self, amount):
        self.balance += amount
        self.available_balance += amount


class Population:
    def __init__(self):
        self.banks = dict()
        self.accounts = dict()

        self.accounts_dataframe = pd.DataFrame(columns=Account.get_dataframe_columns())
        self.bank_to_acc = dict()
        self.role_to_acc = dict()
        self.patterns_to_acc = dict()
        self.compromised = set()

    # GETTERS
    # ------------------------------------------

    def get_bank_ids(self) -> list:
        return list(self.accounts.keys())

    def get_bank_nums(self) -> int:
        return len(self.accounts)

    def get_account(self, account_id):
        return self.accounts[account_id]

    def get_accounts_ids(self) -> list:
        return list(self.accounts.keys())

    def get_accounts_num(self) -> int:
        return len(self.accounts)

    def get_accounts_from_bank(self, bank_id: int) -> list:
        return self.bank_to_acc[bank_id]

    def get_accounts_from_pattern(self, pattern: int) -> list:
        return self.patterns_to_acc[pattern]

    def query_accounts_for_pattern(self, n: int, pattern: int, role: int, node_requirements: dict,
                                   black_list: list) -> dict:
        black_list_extended = [*black_list]
        nodes_dict = dict()

        eligible_accts_df = self.accounts_dataframe
        if pattern is not None:
            eligible_accts_df = eligible_accts_df.iloc[self.patterns_to_acc[pattern]]
        if role is not None:
            eligible_accts_df = eligible_accts_df[eligible_accts_df['role'] == role]
        assert len(eligible_accts_df.index) > 0

        query_acct_df = eligible_accts_df
        # For each node requirement
        for node_id, node_requirement in node_requirements.items():
            # For each requirement of a single node requirements
            for key, req in node_requirement.items():
                query_acct_df = query_acct_df[query_acct_df[key] >= req]

            # This is done in order to relax the constraints imposed. If there's no node which respect the constraints,
            # just the last one (amount) is used to select nodes
            if len(query_acct_df.index) == 0:
                query_acct_df = eligible_accts_df
                requirement = node_requirement.get(_c.GENERAL.BALANCE)
                if requirement is not None:
                    query_acct_df = query_acct_df[query_acct_df[_c.GENERAL.AVAILABLE_BALANCE] >= requirement]
            assert len(query_acct_df.index) > 0

            available = set(query_acct_df.index) - set(black_list_extended)
            chosen_node = random.sample(available, k=n)
            nodes_dict[node_id] = chosen_node
            # The chosen node is appended to black_list to avoid being re-chosen
            black_list_extended.append(chosen_node)
            # Update dataframe node available balance to avoid overlapping inconsistent pattern choices
            amount = node_requirement.get(_c.GENERAL.BALANCE)
            if amount is not None:
                self.accounts_dataframe.loc[
                    self.accounts_dataframe['id'] == chosen_node, _c.GENERAL.AVAILABLE_BALANCE] -= amount
                self.accounts[chosen_node].available_balance -= amount

        return nodes_dict

    # MODIFIERS
    # ------------------------------------------

    def add_bank(self, bank: Bank) -> None:
        self.banks[bank.id] = bank

    def add_account(self, account: Account) -> None:
        assert account.bank_id in self.banks.keys(), print(account.bank_id)

        # Add account to general dictionary
        account_id = account.id
        self.accounts[account_id] = account
        self.accounts_dataframe.iloc[account.id] = account.to_dataframe_row()

        # Add account to Bank->Account.id dictionary
        bank_id = account.bank_id
        add_to_dict_of_list(self.bank_to_acc, bank_id, account_id)

        # Add account to Pattern->Account.id dictionary
        patterns = account.behaviours
        for pattern in patterns:
            add_to_dict_of_list(self.patterns_to_acc, pattern, account_id)

    def add_compromised(self, account_id: int) -> None:
        self.compromised.add(account_id)

    def update_accounts_connections(self, fan_in: list, fan_out: list) -> None:
        self.accounts_dataframe['fan_in'] = fan_in
        self.accounts_dataframe['fan_out'] = fan_out

    # CREATION
    # ------------------------------------------

    def initialize_account_dataframe(self):
        self.accounts_dataframe = pd.DataFrame([account.to_dataframe_row() for account in self.accounts],
                                               columns=Account.get_dataframe_columns())

    def create_launderers(self, mode, limiting_quantile=_v.POPULATION.DEF_ML_LIMITING_QUANTILE):
        if mode == _c.POPULATION.LAUNDERER_CREATION_SIMPLE_MODE:
            self.__create_simple_launderers(limiting_quantile)
        else:
            self.__create_structured_launderers(limiting_quantile)

    def __create_simple_launderers(self, limiting_quantile: float) -> int:
        num_accounts = len(self.accounts)
        num_banks = len(self.bank_to_acc)
        launderers_num = int(num_accounts * _v.POPULATION.DEF_ML_LAUNDERERS_RATIO)

        # Create dataframe from accounts. It is done to speed up the computation
        acc_df = self.accounts_dataframe

        # The following operations are used to remove the most active accounts. This is done by first extracting the 0.9 quantile from the avg_fan_out
        # distribution adn then removing the account with higher avg_fan_out. Launderers does not do many transactions in order to not be bothered
        if limiting_quantile > 0.0:
            quantile_df = acc_df[['avg_tx_per_step']]
            tx_out_quant = np.quantile(quantile_df, limiting_quantile, axis=0)
            acc_df.drop(acc_df[acc_df['avg_tx_per_step'] >= tx_out_quant].index, inplace=True)

        launderers_ids = random.sample(self.accounts.keys(), k=launderers_num)
        for i in launderers_ids:
            self.accounts[i].role = _c.ACCOUNT.ML
            add_to_dict_of_list(self.role_to_acc, _c.ACCOUNT.ML_SOURCE, i)
            add_to_dict_of_list(self.role_to_acc, _c.ACCOUNT.ML_LAYER, i)
            add_to_dict_of_list(self.role_to_acc, _c.ACCOUNT.ML_DESTINATION, i)

        num_launderers = sum([len(l) for _, l in self.role_to_acc.items()])
        assert num_launderers in range(int(launderers_num * 0.9), int(launderers_num * 1.1))

        return num_launderers

    def __create_structured_launderers(self, limiting_quantile: float) -> int:
        num_accounts = len(self.accounts)
        num_banks = len(self.bank_to_acc)
        launderers_num = int(num_accounts * _v.POPULATION.DEF_ML_LAUNDERERS_RATIO)
        banks_laundering_ratio = [bank.launderer_ratio for bank in self.banks]

        # Each one is a list which contains, for each bank, the number of source/layer/dest nodes related to that bank
        num_sources = launderers_num * _v.POPULATION.DEF_ML_SOURCES_PERCENTAGE
        source_dist = banks_laundering_ratio * num_sources

        num_layerer = launderers_num * _v.POPULATION.DEF_ML_LAYERER_PERCENTAGE
        layerer_dist = banks_laundering_ratio * num_layerer

        num_destinations = launderers_num * _v.POPULATION.DEF_ML_DESTINATIONS_PERCENTAGE
        destination_dist = banks_laundering_ratio * num_destinations

        # Create dataframe from accounts. It is done to speed up the computation
        acc_df = self.accounts_dataframe

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
                self.accounts[i].role = _c.ACCOUNT.ML_SOURCE
                add_to_dict_of_list(self.role_to_acc, _c.ACCOUNT.ML_SOURCE, i)

            # Set AMLLayer for bank_i
            bank_users_idxs = bank_users_idxs - set(source_ids)
            layer_ids = random.sample(bank_users_idxs, k=layerer_dist[bank_id])
            for i in layer_ids:
                self.accounts[i].role = _c.ACCOUNT.ML_LAYER
                add_to_dict_of_list(self.role_to_acc, _c.ACCOUNT.ML_LAYER, i)

            # Set AMLDestination for bank_i
            bank_users_idxs = bank_users_idxs - set(layer_ids)
            destination_ids = random.sample(bank_users_idxs, k=destination_dist[bank_id])
            for i in destination_ids:
                self.accounts[i].role = _c.ACCOUNT.ML_DESTINATION
                add_to_dict_of_list(self.role_to_acc, _c.ACCOUNT.ML_DESTINATION, i)

        num_launderers = sum([len(l) for _, l in self.role_to_acc.items()])
        assert num_launderers in range(int(launderers_num * 0.9), int(launderers_num * 1.1))

        return num_launderers

    # BEHAVIOUR
    # ------------------------------------------
    def perform_cash_tx(self, account_id: int, amount: float) -> None:
        account = self.accounts[account_id]
        account.update_balance(+amount)
        self.accounts_dataframe.at[account_id, 'balance'] += amount
        self.accounts_dataframe.at[account_id, 'available_balance'] += amount

    def send_transaction(self, originator_id: int, beneficiary_id: int, amount: int) -> bool:
        originator = self.accounts[originator_id]
        beneficiary = self.accounts[beneficiary_id]

        outcome = originator.has_amount()
        if outcome:
            originator.update_balance(-amount)
            self.accounts_dataframe.at[originator_id, 'balance'] -= amount
            beneficiary.update_balance(amount)
            self.accounts_dataframe.at[originator_id, 'balance'] += amount

        return outcome

    def get_compromised(self, num_compromised: int) -> set:
        # Get compromised accounts based on compromised probability
        compromised = np.random.choice(len(self.accounts), size=num_compromised, replace=False,
                                       p=self.accounts_dataframe['compromising_ratio'])
        # Check whether some chosen account has already been compromised in the past...
        already_compromised = [acc in compromised for acc in self.compromised]
        # ...if so add those accounts to a 'to remove' list...
        to_remove = already_compromised

        # Keep choosing accounts until only available accounts are found
        while len(already_compromised) > 0:
            compromised = np.random.choice(len(self.accounts), size=num_compromised, replace=False,
                                           p=self.accounts_dataframe['compromising_ratio'])
            already_compromised = [acc in compromised for acc in self.compromised]
            to_remove.extend(already_compromised)

        # ...and then remove them. They have used the opportunity to be not compromised a second time
        self.compromised = self.compromised - set(to_remove)

        self.compromised = set.union(self.compromised, set(compromised))

        return compromised
