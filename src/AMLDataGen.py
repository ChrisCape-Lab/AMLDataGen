import os
import random
import numpy as np
import yaml

import pandas as pd
import networkx as nx

from src._constants import *
from src.model.population import Population, Account, Bank, Community
from src.model.pattern import GeneralPattern, StructuredPattern



class AMLDataGen:
    def __init__(self, _conf_file):
        # Open config file
        with open(_conf_file, "r") as rf:
            self.conf = yaml.safe_load(rf)

        # Entities
        self.banks = dict()
        self.population = Population()
        self.num_accounts = 0
        self.num_launderers = 0

        # Connection Graph
        self.connection_graph = nx.DiGraph()

        # Patterns
        self.normal_patterns = dict()
        self.ml_patterns = dict()

        # Simulator Data
        simulation_conf = self.conf['Simulation']
        self.end_time = simulation_conf['end_time']
        self.transactions = pd.DataFrame(columns=['Originator', 'Beneficiary', 'Amount', 'Time', 'Type'])

        # Output files
        input_files = self.conf['Output_files']
        self.accounts_out_file = input_files['account_file']
        self.transaction_out_file = input_files['transaction_file']

        # Queues
        # TODO: the idea is a dict where key is start_time and item is patterns with that start time. Each time a pattern is executed, the queue is updated with the current time
        self.normal_queue = dict()
        self.aml_queue = dict()


    # LOADERS
    # ------------------------------------------

    def load_all(self):
        input_files = self.conf['Input_files']

        accounts_file = input_files['account_file']
        self.load_accounts(accounts_file)

        bank_file = input_files['bank_file']
        self.load_banks(bank_file)

        communities_file = input_files['communities_file']
        self.load_communities(communities_file)

        normal_pattern_file = input_files['normal_pattern_file']
        self.load_normal_patterns(normal_pattern_file)

        aml_pattern_file = input_files['aml_pattern_file']
        self.load_aml_patterns(aml_pattern_file)


    def load_accounts(self, accounts_file):
        acc_df = pd.read_csv(accounts_file)
        acct_id = 0
        for _, row in acc_df.iterrows():
            for _ in row['count']:
                balance = random.uniform(row['balance_min'], row['balance_max'])
                business = row['business']
                behaviours = row['behaviours']
                bank_id = row['bank_id']
                community = None
                avg_fan_in = random.uniform(row['fan_in_min'], row['fan_in_max'])
                avg_fan_out = random.uniform(row['fan_out_min'], row['fan_out_max'])
                min_amount = random.gauss(row['min_amount'], row['min_amount']/6)
                max_amount = random.gauss(row['max_amount'], row['max_amount']/6)
                avg_amount = random.gauss(row['avg_amount'], row['avg_amount']/6)
                compromising_ratio = random.uniform(row['compromising_ratio']-0.1, row['compromising_ratio']+0.1)
                role = NORMAL

                account = Account(acct_id, balance, business, behaviours, bank_id, community, avg_fan_in, avg_fan_out, min_amount, max_amount, avg_amount,
                                  compromising_ratio, role)

                self.population.add_account(account)
                acct_id += 1


    def load_banks(self, banks_file):
        bank_df = pd.read_csv(banks_file)
        bank_id = 0
        for _, row in bank_df.iterrows():
            self.banks[bank_id] = Bank(bank_id, row['compromising_ratio'], row['launderer_ratio'])
            bank_id += 1

        assert sum([bank.launderer_ratio for _, bank in self.banks.items()]) == 1

    def create_launderers(self):

        for _, bank in self.banks.items()

        self.num_launderers = self.population.create_launderers()

    def load_normal_patterns(self, normal_patterns_file):
        normal_patterns_df = pd.read_csv(normal_patterns_file)
        normal_patterns_id = 0
        for _, row in normal_patterns_df.iterrows():
            pattern_type = row['pattern_type']
            period = row['period']
            min_amount = row['min_amount']
            max_amount = row['max_amount']
            min_accounts = row['min_accounts']
            max_accounts = row['max_accounts']
            scheduling = row['scheduling']
            for _ in row['count']:
                amount = random.uniform(min_amount, max_amount)
                accounts_num = random.randint(min_accounts, max_accounts)
                accounts = random.sample(self.population.get_accounts_ids(), k=accounts_num)
                pattern = GeneralPattern(normal_patterns_id, pattern_type, period, amount, accounts, scheduling_type=scheduling)

                pattern.schedule(self.end_time)

                start_time = min(pattern.scheduling_times)
                if start_time not in self.normal_queue.keys():
                    self.normal_queue[start_time] = [pattern]
                else:
                    self.normal_queue[start_time].append(pattern)

    def load_aml_patterns(self, aml_patterns_file):
        aml_patterns_df = pd.read_csv(aml_patterns_file)
        aml_patterns_id = 0
        for _, row in aml_patterns_df.iterrows():
            pattern_type = row['pattern_type']
            period = row['period']
            min_amount = row['min_amount']
            max_amount = row['max_amount']
            min_accounts = row['min_accounts']
            max_accounts = row['max_accounts']
            scheduling = row['scheduling']
            for _ in row['count']:
                amount = random.uniform(min_amount, max_amount)
                accounts_num = random.uniform(min_accounts, max_accounts)

                pattern = StrcturedPattern(aml_patterns_id, pattern_type, period, amount, scheduling_type=scheduling)

                accounts_dist = pattern.get_account_distribution(accounts_num)
                # TODO: maybe a FIFO queue is better to distribute the roles?
                sources = random.sample(self.population.aml_sources, accounts_dist[0])
                pattern.add_source(sources)
                layers = random.sample(self.population.aml_layerer, accounts_dist[1])
                pattern.add_layerer(layers)
                destinations = random.sample(self.population.aml_destinations, accounts_dist[2])
                pattern.add_destination(destinations)

                pattern.schedule(self.end_time)

                start_time = min(pattern.scheduling_times)
                if start_time not in self.aml_queue.keys():
                    self.aml_queue[start_time] = [pattern]
                else:
                    self.aml_queue[start_time].append(pattern)

    def schedule_normal_transactions(self):
        cash_in_requests = set()
        for t in range(0, self.end_time):
            for acc_id in range(0, self.population.get_accounts_num()):
                account = self.population.get_accounts()[acc_id]

                # Do a cash-in for requested users
                if acc_id in cash_in_requests:
                    self._schedule_cash_in(acc_id, t)

                # Determine how many transactions a user can do in the instant
                acc_avg_fan_out = account.avg_fan_out
                min_txs_number = min(int(acc_avg_fan_out * 0.85), acc_avg_fan_out-1)
                max_txs_number = max(int(acc_avg_fan_out * 1.15), acc_avg_fan_out+1)
                tx_number = random.randint(min_txs_number, max_txs_number)
                for i in range(0, tx_number):
                    req_cash_in = self._schedule_account_tx(acc_id, t)
                    if req_cash_in:
                        cash_in_requests.add(req_cash_in)
                        break
                if account.balance


    def _schedule_cash_in(self, account_id, time):
        account = self.population.get_accounts()[account_id]
        numbers = random.randint(1, 3)
        for i in range(0, numbers):
            amount = account.max_amount * random.randint(2, 5-numbers)
            account.update(None, amount)
            self.transactions.append((account_id, None, amount, time, 'Normal'))

    def _schedule_account_tx(self, account_id, time):
        originator = self.population.get_accounts()[account_id]
        beneficiary_id, amount = originator.get_transaction()

        # If beneficiary is -1 means that accounts require a new beneficiary
        if beneficiary_id == -1:
            available_nodes = list(self.population.get_accounts_ids().copy())
            available_nodes.remove(originator.id)
            beneficiary_id = random.choice(available_nodes)

        outcome = originator.update(beneficiary_id, amount)
        if outcome:
            self.population.get_accounts()[beneficiary_id].update(originator.id, -amount)
            self.transactions.append((originator.id, beneficiary_id, -amount, time, 'Normal'))

        return outcome

    def schedule_normal_patterns(self):
        for patterns in self.normal_patterns.items():
            for pattern in patterns:
                pattern.schedule_tx()
                for orig, bene, amt, time, tx_type in pattern.get_transactions():
                    outcome = self.population.get_accounts()[orig].update(None, -amt)
                    if outcome:
                        self.population.get_accounts()[bene].update(None, amt)
                        self.transactions.append((orig, bene, amt, time, tx_type))

    def schedule_aml_patterns(self):
        for patterns in self.ml_patterns.items():
            for pattern in patterns:
                pattern.schedule_tx()
                for orig, bene, amt, time, tx_type in pattern.get_transactions():
                    outcome = self.population.get_accounts()[orig].update(None, -amt)
                    if outcome:
                        self.population.get_accounts()[bene].update(None, amt)
                        self.transactions.append((orig, bene, amt, time, tx_type))








