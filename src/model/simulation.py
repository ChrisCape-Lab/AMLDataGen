import logging
import random
from datetime import datetime

from src.model.population import Population
from src.model.community import Community
from src.model.datawriter import DataWriter
from src.model.pattern import Pattern
from src.utils import add_to_dict_of_list

import src._constants as _c
import src._variables as _v

"""
def handle_requirements(df: pd.DataFrame, role: int, n: int, node_requirements: dict, black_list: list) -> dict:
    black_list_extended = [*black_list]
    nodes_dict = dict()
    # For each node requirement
    for node_id, node_requirement in node_requirements.items():
        # For each requirement of a single node requirements
        dataframe = df[df['role'] == role]
        for key, req in node_requirement.items():
            dataframe = dataframe[dataframe[key] >= req]

        # This is done in order to relax the constraints imposed. If there's no node which respect the constraints,
        # just the last one (amount) is used to select nodes
        if len(dataframe.index) == 0:
            dataframe = df[df['role'] == role]
            requirement = node_requirement.get(GENERAL.BALANCE)
            if requirement is not None:
                dataframe = dataframe[dataframe[GENERAL.AVAILABLE_BALANCE] >= requirement]

        available = set(dataframe.index) - set(black_list_extended)
        chosen_node = random.sample(available, k=n)
        nodes_dict[node_id] = chosen_node
        # The chosen node is appended to black_list to avoid being re-chosen
        black_list_extended.append(chosen_node)
        # Update dataframe node available balance to avoid overlapping inconsistent pattern choices
        df.loc[df['id'] == chosen_node, GENERAL.AVAILABLE_BALANCE] += df[GENERAL.AVAILABLE_BALANCE]

    return nodes_dict
"""

class Simulation:
    def __init__(self, population: Population, community: Community, datawriter: DataWriter, start_time: int,
                 end_time: int):
        self.population = population
        self.community = community
        self.datawriter = datawriter
        self.start_time = start_time
        self.end_time = end_time
        self.normal_patterns_to_schedule = dict()
        self.ml_patterns_to_schedule = dict()
        self.scheduled_patterns = list()

        self.tx_id = 0
        self.transaction_list = list()

        self.allow_random_tx = True

        self.time = start_time

    # SETTERS
    # ------------------------------------------

    def load_normal_patterns(self, patterns: list) -> None:
        for pattern in patterns:
            time = random.randint(self.start_time, self.end_time)
            pattern.create_structure()
            add_to_dict_of_list(self.normal_patterns_to_schedule, time, pattern)

    def load_ml_patterns(self, patterns: list) -> None:
        for pattern in patterns:
            time = random.randint(self.start_time, self.end_time)
            pattern.create_structure()
            add_to_dict_of_list(self.ml_patterns_to_schedule, time, pattern)

    # BEHAVIOUR
    # ------------------------------------------

    def setup(self, allow_random_txs: bool) -> None:
        self.allow_random_tx = allow_random_txs
        self.population.create_launderers(_v.SIMULATION.DEF_LAUNDERERS_CREATION_MODE)
        self.community.create_community(community_type=_v.COMMUNITY.DEF_COMMUNITY_TYPE)
        self.__handle_new_connections()

    def run(self) -> None:
        for t in range(self.start_time, self.end_time):
            self.step(t)

    def step(self, time: int) -> None:
        start = datetime.now().replace(microsecond=0)
        self.__handle_accounts_cashes(time)
        self.__handle_patterns_transactions(time)
        self.__handle_patterns_to_schedule(time)
        if self.allow_random_tx:
            self.__handle_accounts_random_txs(time)
        self.__handle_new_connections()

        out = "  - [" + str(datetime.now().replace(microsecond=0)-start) + "] Step " + str(self.time + 1) + "/" + str(self.end_time) + ": Done"
        print(out)
        logging.info(out)
        self.time += 1

    def __handle_accounts_cashes(self, time: int) -> None:
        accounts = self.population.accounts.values()
        for account in accounts:
            # Handle cash-in requests
            if account.require_cash_in():
                for _ in range(0, account.get_number_of_cash_tx()):
                    amount = account.get_cash_in_amount()
                    transaction = (self.tx_id, None, account.id, amount, time, _c.GENERAL.RANDOM, False)
                    self.population.perform_cash_tx(account.id, amount)
                    self.transaction_list.append(transaction)
                    self.__check_flush_tx_to_file()
                    self.tx_id += 1

            # Handle cash-out requests
            if account.require_cash_out():
                for _ in range(0, account.get_number_of_cash_tx()):
                    amount = account.get_cash_out_amount()
                    transaction = (self.tx_id, account.id, None, amount, time, _c.GENERAL.RANDOM, False)
                    self.population.perform_cash_tx(account.id, -amount)
                    self.transaction_list.append(transaction)
                    self.__check_flush_tx_to_file()
                    self.tx_id += 1

    def __handle_patterns_to_schedule(self, time: int) -> None:

        def schedule_pattern(pattern: Pattern, is_normal: bool) -> None:
            pattern_type = pattern.pattern_type

            # Get the available sources from the dataframe
            num_sources, source_req = pattern.get_sources_requirements()
            role = _c.ACCOUNT.NORMAL if is_normal else _c.ACCOUNT.ML_SOURCE
            sources_map = self.population.query_accounts_for_pattern(pattern_type, role, source_req, [])
            pattern.add_nodes(sources_map)
            sources_list = list(sources_map.values())

            # Get the available layerers from the dataframe
            num_layerer, layerer_req = pattern.get_layerer_requirements()
            role = _c.ACCOUNT.NORMAL if is_normal else _c.ACCOUNT.ML_LAYER
            layerers_map = self.population.query_accounts_for_pattern(pattern_type, role, layerer_req, sources_list)
            pattern.add_nodes(layerers_map)
            layerers_list = list(layerers_map.values())

            # Get the available destination from the dataframe
            num_destinations, destination_req = pattern.get_destinations_requirements()
            role = _c.ACCOUNT.NORMAL if is_normal else _c.ACCOUNT.ML_DESTINATION
            destinations_map = self.population.query_accounts_for_pattern(pattern_type, role, destination_req,
                                                                          [*sources_list, *layerers_list])
            pattern.add_nodes(destinations_map)

            # After all nodes are added to the pattern, the transactions can be scheduled
            pattern.schedule(time)

        ml_patterns_to_schedule = self.ml_patterns_to_schedule.get(time, list())
        for pattern in ml_patterns_to_schedule:
            schedule_pattern(pattern, is_normal=False)
            ml_patterns_to_schedule.remove(pattern)
            self.scheduled_patterns.append(pattern)

        normal_patterns_to_schedule = self.normal_patterns_to_schedule.get(time, list())
        for pattern in normal_patterns_to_schedule:
            schedule_pattern(pattern, is_normal=True)
            normal_patterns_to_schedule.remove(pattern)
            self.scheduled_patterns.append(pattern)

    def __handle_patterns_transactions(self, time: int) -> None:
        for pattern in self.scheduled_patterns:
            txs_to_schedule = pattern.schedule_txs(time)

            if txs_to_schedule is None:
                continue

            for (src, dst, amt, time, tx_type) in txs_to_schedule:
                self.__execute_transaction(src, dst, amt, time, tx_type, pattern.is_aml)

            self.__check_flush_tx_to_file()

    def __handle_accounts_random_txs(self, time: int) -> None:
        accounts = self.population.accounts
        for _, account in accounts.items():
            # Determine whether to perform a completely random cash_in
            cash_in_probability = random.random() > _v.ACCOUNT.DEF_CASH_IN_PROB
            if cash_in_probability and not account.require_cash_out:
                amount = account.get_cash_in_amount()
                transaction = (self.tx_id, None, account.id, amount, time, _c.GENERAL.RANDOM, False)
                self.population.perform_cash_tx(account.id, amount)
                self.transaction_list.append(transaction)
                self.tx_id += 1

            # Determine whether to perform a completely random cash_out
            cash_out_probability = random.random() > _v.ACCOUNT.DEF_CASH_OUT_PROB
            if cash_out_probability and not account.require_cash_in():
                amount = account.get_cash_out_amount()
                if amount > 0:
                    transaction = (self.tx_id, account.id, None, amount, time, _c.GENERAL.RANDOM, False)
                    self.population.perform_cash_tx(account.id, -amount)
                    self.transaction_list.append(transaction)
                    self.tx_id += 1

            # Execute the transactions for the user
            for _ in range(0, account.get_number_of_txs()):
                beneficiary_id = self.community.get_random_destination_for(account.id, account.new_beneficiary_ratio)
                amount = account.get_tx_amount()

                self.__execute_transaction(account.id, beneficiary_id, amount, time, _c.GENERAL.RANDOM, False)

            self.__check_flush_tx_to_file()

    def __handle_new_connections(self):
        fan_in_list = self.community.get_fan_in_list()
        fan_out_list = self.community.get_fan_out_list()
        self.population.update_accounts_connections(fan_in_list, fan_out_list)

    def __execute_transaction(self, src, dst, amt, time, tx_type, is_aml):
        transaction = (self.tx_id, src, dst, amt, time, tx_type, is_aml)
        outcome = self.population.send_transaction(src, dst, amt, tx_type)
        if outcome:
            self.transaction_list.append(transaction)
            self.community.add_link(src, dst)
            self.tx_id += 1
        else:
            balance = self.population.accounts[src].balance
            if tx_type in [_c.GENERAL.CYCLE, _c.GENERAL.U, _c.GENERAL.SCATTER_GATHER]:
                logging.critical("Missed transaction on user " + str(src) + " : balance = " + str(balance) + "and amount = " + str(amt)
                                 + " on pattern " + str(tx_type))
            elif tx_type == _c.GENERAL.BIPARTITE:
                logging.error("Missed transaction on user " + str(src) + " : balance = " + str(balance) + "and amount = " + str(amt) + " on pattern "
                              + str(tx_type))
            else:
                logging.warning("Missed transaction on user " + str(src) + " : balance = " + str(balance) + "and amount = " + str(amt)
                                + " on pattern " + str(tx_type))

    def __check_flush_tx_to_file(self):
        if len(self.transaction_list) >= 10000:
            self.datawriter.write_transactions(self.transaction_list)
            self.transaction_list.clear()






