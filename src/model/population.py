import random

import networkx as nx
import numpy as np
import pandas as pd
import logging

import src._constants as _c
import src._variables as _v

from src.model.entities import Bank, Account
from src.model.community import Community
from src.utils import add_to_dict_of_list


class Population:
    def __init__(self):
        # Entities
        self.__banks = dict()
        self.__accounts = dict()

        # Population data
        self.__accounts_dataframe = self.__create_accounts_dataframe()
        self.__community = Community()

        # Access structures
        self.__bank_to_acc = dict()
        self.__role_to_acc = dict()
        self.__patterns_to_acc = dict()
        self.__compromised = set()

    # INITIATORS
    # ------------------------------------------
    @staticmethod
    def __create_accounts_dataframe():
        df = pd.DataFrame(columns=Account.get_dataframe_columns())
        df = df.astype(Account.get_dataframe_column_type())
        df.set_index("id", inplace=True)

        return df

    # GETTERS
    # ------------------------------------------

    def get_bank_ids(self) -> list:
        return list(self.__banks.keys())

    def get_account(self, account_id):
        return self.__accounts[account_id]

    def get_accounts_ids(self) -> list:
        return list(self.__accounts.keys())

    def get_accounts_from_bank(self, bank_id: int) -> list:
        return self.__bank_to_acc[bank_id]

    def get_accounts_from_pattern(self, pattern: int) -> list:
        return self.__patterns_to_acc[pattern]

    def get_random_destination_for(self, account_id: int) -> int:
        account = self.__accounts[account_id]
        return self.__community.get_random_destination_for(account.id, account.new_beneficiary_ratio)

    # QUERIES
    # ------------------------------------------

    def require_cash_in(self, account_id: int) -> bool:
        return self.__accounts[account_id].require_cash_in()

    def require_cash_out(self, account_id: int) -> bool:
        return self.__accounts[account_id].require_cash_out()

    def get_number_cash_tx_of(self, account_id: int) -> int:
        return self.__accounts[account_id].get_number_of_cash_tx()

    def get_cash_in_amount_for(self, account_id: int) -> float:
        return self.__accounts[account_id].get_cash_in_amount()

    def get_cash_out_amount_for(self, account_id: int) -> float:
        return self.__accounts[account_id].get_cash_out_amount()

    def get_number_of_txs_for(self, account_id: int) -> int:
        return self.__accounts[account_id].get_number_of_txs()

    def get_tx_amount_for(self, account_id: int) -> int:
        return self.__accounts[account_id].get_tx_amount()

    def query_accounts_with_relations(self, pattern: int, pattern_graph: nx.Graph) -> dict:
        available_nodes = filter(lambda (n, d): pattern in d['patterns'], self.__community.connection_graph.nodes(data=True))
        community_subgraph = self.__community.connection_graph.subgraph(available_nodes)

        def attribute_match_function(comm_node, pttrn_node):
            return comm_node["capacity"] >= pttrn_node["available_balance"]

        gm = GraphMatcher(community_subgraph, pattern_graph)
        mapping = gm.mapping()

        pattern_capacity_dict = nx.get_node_attributes(pattern_graph, "capacity")
        for community_node, pattern_node in mapping.items():
            amount = pattern_capacity_dict[pattern_node]
            self.__community.nodes[community_node].capacity -= amount
            self.__accounts[community_node].update_available_balance(-amount)
            self.__accounts_dataframe.at[community_node, _c.GENERAL.AVAILABLE_BALANCE] -= amount

        return mapping

    def query_accounts_without_relations(self, pattern: int, role: int, node_requirements: list,
                                         black_list: list) -> dict:
        black_list_extended = [*black_list]
        nodes_dict = dict()

        eligible_accts_df = self.__accounts_dataframe
        if pattern is not None:
            eligible_accts_df = self.__accounts_dataframe[self.__accounts_dataframe.index.isin(self.__patterns_to_acc[pattern])]
        if role is not None:
            eligible_accts_df_tmp = eligible_accts_df[eligible_accts_df['role'] == role]
            # This is done in order to avoid the emptiness due to the simple form: in this case the df contains only
            # ML as entry, not a specific role (SOURCE, LAYERER, DESTINATION) and in this case all are good. IF this is
            # the situation, the role check is just bypassed
            if eligible_accts_df_tmp.empty:
                # If the previous check returns empty means that we are in an unstructured scenario. Thus, every
                # launderer can be whatever ml role
                eligible_accts_df = eligible_accts_df[eligible_accts_df['role'] != _c.ACCOUNT.NORMAL]
            else:
                # Otherwise, just assign the tmp variable to the correct one
                eligible_accts_df = eligible_accts_df_tmp
        assert len(eligible_accts_df.index) > 0

        query_acct_df = eligible_accts_df
        # For each node requirement
        for requirement in node_requirements:
            # For each requirement of a single node requirements
            for key in requirement.get_requirements_keys():
                query_acct_df = query_acct_df.query(requirement.get_requirement_as_condition(key))

            # This is done in order to relax the constraints imposed. If there's no node which respect the constraints,
            # just the last one (amount) is used to select nodes
            if len(query_acct_df.index) == 0:
                query_acct_df = eligible_accts_df
                query_acct_df = query_acct_df.query(requirement.get_requirement_as_condition(_c.GENERAL.AVAILABLE_BALANCE))
                # Log the missing requirements check
                out_str = "Too restrictive requirements for pattern " + str(pattern) + " : "
                for key in requirement.get_requirements_keys():
                    out_str += ", " + requirement.get_requirement_as_condition(key)
                logging.warning(out_str)
            assert len(query_acct_df.index) > 0

            available = set(query_acct_df.index.tolist()) - set(black_list_extended)
            chosen_node = random.sample(available, k=1)[0]
            nodes_dict[requirement.get_node_id_to_map()] = chosen_node
            # The chosen node is appended to black_list to avoid being re-chosen
            black_list_extended.append(chosen_node)
            # Update dataframe node available balance to avoid overlapping inconsistent pattern choices
            amount = requirement.get_requirement_value(_c.GENERAL.AVAILABLE_BALANCE)

            if amount is not None and amount != 0:
                if self.__accounts[chosen_node].available_balance != self.__accounts_dataframe.loc[chosen_node][_c.GENERAL.AVAILABLE_BALANCE]:
                    print("Hey: class is " + str(self.__accounts[chosen_node].available_balance) + " but dataframe is " +
                          str(self.__accounts_dataframe.loc[chosen_node][_c.GENERAL.AVAILABLE_BALANCE]))
                self.__accounts[chosen_node].update_available_balance(-amount)
                self.__accounts_dataframe.at[chosen_node, _c.GENERAL.AVAILABLE_BALANCE] -= amount

        return nodes_dict

    # MODIFIERS
    # ------------------------------------------

    def add_bank(self, bank: Bank) -> None:
        self.__banks[bank.id] = bank

    def add_account(self, account: Account) -> None:
        assert account.bank_id in self.__banks.keys(), print(account.bank_id)

        # Add account to general dictionary
        account_id = account.id
        self.__accounts[account_id] = account

        # Add to population data
        self.__accounts_dataframe.loc[account_id] = account.to_dataframe_row()[1:]
        self.__community.add_node(account)
        assert len(self.__accounts) == len(self.__accounts_dataframe.index)

        # Add account to Bank->Account.id dictionary
        bank_id = account.bank_id
        add_to_dict_of_list(self.__bank_to_acc, bank_id, account_id)

        # Add account to Pattern->Account.id dictionary
        patterns = account.behaviours
        for pattern in patterns:
            add_to_dict_of_list(self.__patterns_to_acc, pattern, account_id)

    def add_compromised(self, account_id: int) -> None:
        self.__compromised.add(account_id)

    #TODO: move it into the send transaction and community creation
    def update_accounts_connections(self):
        fan_in_list = [len(x) for x in self.__community.get_fan_in_list()]
        fan_out_list = [len(x) for x in self.__community.get_fan_out_list()]

        assert len(fan_in_list) == len(fan_out_list) == len(self.__accounts)

        self.__accounts_dataframe[_c.GENERAL.FAN_IN_NAME] = fan_in_list
        self.__accounts_dataframe[_c.GENERAL.FAN_OUT_NAME] = fan_out_list

    # CREATION
    # ------------------------------------------

    def create_community(self, community_type: int = _v.COMMUNITY.DEF_COMMUNITY_TYPE):
        self.__community.create_community(community_type=community_type)

    def create_launderers(self, mode, limiting_quantile=_v.POPULATION.DEF_ML_LIMITING_QUANTILE):
        if mode == _c.POPULATION.LAUNDERER_CREATION_SIMPLE_MODE:
            self.__create_simple_launderers(limiting_quantile)
        else:
            self.__create_structured_launderers(limiting_quantile)

    def __create_simple_launderers(self, limiting_quantile: float) -> int:
        num_accounts = len(self.__accounts)
        num_banks = len(self.__bank_to_acc)
        launderers_num = int(num_accounts * _v.POPULATION.DEF_ML_LAUNDERERS_RATIO)

        # Create dataframe from accounts. It is done to speed up the computation
        acc_df = self.__accounts_dataframe

        # The following operations are used to remove the most active accounts. This is done by first extracting the 0.9 quantile from the avg_fan_out
        # distribution adn then removing the account with higher avg_fan_out. Launderers does not do many transactions in order to not be bothered
        if limiting_quantile > 0.0:
            quantile_df = acc_df[['avg_tx_per_step']]
            tx_out_quant = np.quantile(quantile_df, limiting_quantile, axis=0)[0]
            acc_df = acc_df.drop(acc_df[acc_df['avg_tx_per_step'] >= tx_out_quant].index.tolist())

        launderers_ids = random.sample(acc_df.index.tolist(), k=launderers_num)
        for i in launderers_ids:
            self.__accounts[i].role = _c.ACCOUNT.ML
            self.__accounts_dataframe.at[i, 'role'] = _c.ACCOUNT.ML
            add_to_dict_of_list(self.__role_to_acc, _c.ACCOUNT.ML, i)

        num_launderers = sum([len(l) for _, l in self.__role_to_acc.items()])
        assert num_launderers in range(int(launderers_num * 0.9), int(launderers_num * 1.1))

        return num_launderers

    def __create_structured_launderers(self, limiting_quantile: float) -> int:
        num_accounts = len(self.__accounts)
        num_banks = len(self.__bank_to_acc)
        launderers_num = int(num_accounts * _v.POPULATION.DEF_ML_LAUNDERERS_RATIO)
        banks_laundering_ratio = [bank.launderer_ratio for bank in self.__banks]

        # Each one is a list which contains, for each bank, the number of source/layer/dest nodes related to that bank
        num_sources = launderers_num * _v.POPULATION.DEF_ML_SOURCES_PERCENTAGE
        source_dist = banks_laundering_ratio * num_sources

        num_layerer = launderers_num * _v.POPULATION.DEF_ML_LAYERER_PERCENTAGE
        layerer_dist = banks_laundering_ratio * num_layerer

        num_destinations = launderers_num * _v.POPULATION.DEF_ML_DESTINATIONS_PERCENTAGE
        destination_dist = banks_laundering_ratio * num_destinations

        # Create dataframe from accounts. It is done to speed up the computation
        acc_df = self.__accounts_dataframe

        # The following operations are used to remove the most active accounts. This is done by first extracting the 0.9 quantile from the avg_fan_out
        # distribution adn then removing the account with higher avg_fan_out. Launderers does not do many transactions in order to not be bothered
        if limiting_quantile > 0.0:
            quantile_df = acc_df[['avg_tx_per_step']]
            tx_out_quant = np.quantile(quantile_df, limiting_quantile, axis=0)
            acc_df = acc_df.drop(acc_df[acc_df['avg_tx_per_step'] >= tx_out_quant].index.tolist())

        # For each bank are extracted sources, layers and destinations
        for bank_id in range(0, num_banks):
            bank_users_idxs = set(acc_df[acc_df[[5] == bank_id]][[0]])

            # Set AMLSources for bank_i
            source_ids = random.sample(bank_users_idxs, k=source_dist[bank_id])
            for i in source_ids:
                self.__accounts[i].role = _c.ACCOUNT.ML_SOURCE
                self.__accounts_dataframe.at[i, 'role'] = _c.ACCOUNT.ML_SOURCE
                add_to_dict_of_list(self.__role_to_acc, _c.ACCOUNT.ML_SOURCE, i)

            # Set AMLLayer for bank_i
            bank_users_idxs = bank_users_idxs - set(source_ids)
            layer_ids = random.sample(bank_users_idxs, k=layerer_dist[bank_id])
            for i in layer_ids:
                self.__accounts[i].role = _c.ACCOUNT.ML_LAYER
                self.__accounts_dataframe.at[i, 'role'] = _c.ACCOUNT.ML_LAYER
                add_to_dict_of_list(self.__role_to_acc, _c.ACCOUNT.ML_LAYER, i)

            # Set AMLDestination for bank_i
            bank_users_idxs = bank_users_idxs - set(layer_ids)
            destination_ids = random.sample(bank_users_idxs, k=destination_dist[bank_id])
            for i in destination_ids:
                self.__accounts[i].role = _c.ACCOUNT.ML_DESTINATION
                self.__accounts_dataframe.at[i, 'role'] = _c.ACCOUNT.ML_DESTINATION
                add_to_dict_of_list(self.__role_to_acc, _c.ACCOUNT.ML_DESTINATION, i)

        num_launderers = sum([len(l) for _, l in self.__role_to_acc.items()])
        assert num_launderers in range(int(launderers_num * 0.9), int(launderers_num * 1.1))

        return num_launderers

    # BEHAVIOUR
    # ------------------------------------------

    def perform_cash_in_tx(self, account_id: int, amount: float) -> bool:
        assert amount > 0

        account = self.__accounts[account_id]

        account.update_balance(amount)
        self.__accounts_dataframe.at[account_id, 'balance'] += amount

        account.update_available_balance(amount)
        self.__accounts_dataframe.at[account_id, 'available_balance'] += amount

        return True

    def perform_cash_out_tx(self, account_id: int, amount: float) -> bool:
        assert amount > 0
        account = self.__accounts[account_id]
        outcome = account.has_amount(amount)
        if outcome:
            account.update_balance(-amount)
            self.__accounts_dataframe.at[account_id, 'balance'] -= amount

            account.update_available_balance(-amount)
            self.__accounts_dataframe.at[account_id, 'available_balance'] -= amount

        return outcome

    def send_transaction(self, originator_id: int, beneficiary_id: int, amount: float, pattern: int) -> bool:
        assert amount > 0
        originator = self.__accounts[originator_id]
        beneficiary = self.__accounts[beneficiary_id]

        outcome = originator.has_amount(amount)
        if outcome:
            originator.update_balance(-amount)
            self.__accounts_dataframe.at[originator_id, 'balance'] -= amount

            beneficiary.update_balance(amount)
            self.__accounts_dataframe.at[beneficiary_id, 'balance'] += amount

            self.__community.add_link(originator_id, beneficiary_id)

            # If the pattern is NOT random, the available balance was yet removed by the query function
            if pattern == _c.GENERAL.RANDOM:
                originator.update_available_balance(-amount)
                self.__accounts_dataframe.at[originator_id, _c.GENERAL.AVAILABLE_BALANCE] -= amount

                beneficiary.update_available_balance(amount)
                self.__accounts_dataframe.at[beneficiary_id, _c.GENERAL.AVAILABLE_BALANCE] += amount

        # If transaction is refused AND the pattern is not random, the users must regain the spent available balance
        elif pattern != _c.GENERAL.RANDOM:
            originator.update_available_balance(amount)
            self.__accounts_dataframe.at[originator_id, _c.GENERAL.AVAILABLE_BALANCE] += amount

            beneficiary.update_available_balance(-amount)
            self.__accounts_dataframe.at[beneficiary_id, _c.GENERAL.AVAILABLE_BALANCE] -= amount

        return outcome

    def get_compromised(self, num_compromised: int) -> set:
        # Get compromised accounts based on compromised probability
        compromised = np.random.choice(len(self.__accounts), size=num_compromised, replace=False,
                                       p=self.__accounts_dataframe['compromising_ratio'])
        # Check whether some chosen account has already been compromised in the past...
        already_compromised = [acc in compromised for acc in self.__compromised]
        # ...if so add those accounts to a 'to remove' list...
        to_remove = already_compromised

        # Keep choosing accounts until only available accounts are found
        while len(already_compromised) > 0:
            compromised = np.random.choice(len(self.__accounts), size=num_compromised, replace=False,
                                           p=self.__accounts_dataframe['compromising_ratio'])
            already_compromised = [acc in compromised for acc in self.__compromised]
            to_remove.extend(already_compromised)

        # ...and then remove them. They have used the opportunity to be not compromised a second time
        self.__compromised = self.__compromised - set(to_remove)

        self.__compromised = set.union(self.__compromised, set(compromised))

        return compromised
