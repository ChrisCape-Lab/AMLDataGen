import os
import random
import numpy as np
import yaml

import pandas as pd
import networkx as nx

from src.model import *



class AMLDataGen:
    def __init__(self, _conf_file):
        # Open config file
        with open(_conf_file, "r") as rf:
            self.conf = yaml.safe_load(rf)

        # Connection Graph
        self.g = nx.DiGraph()

        # Entities
        self.num_accounts = 0
        self.accounts = dict()
        self.num_launderers = 0
        self.aml_sources = []
        self.aml_layerer = []
        self.aml_destinations = []
        self.banks = dict()
        self.normal_patterns = dict()
        self.ml_patterns = dict()

        # Input files
        accounts_file =
        self.load_accounts(accounts_file)

        # Queues
        # TODO: the idea is a dict where key is start_time and item is patterns with that start time. Each time a pattern is executed, the queue is updated with the current time
        self.normal_queue = dict()
        self.aml_queue = dict()



    def load_accounts(self, accounts_file):
        acc_df = pd.read_csv(accounts_file)
        acct_id = 0
        for _, row in acc_df.iterrows():
            for _ in row['count']:
                balance = random.uniform(row['balance_min'], row['balance_max'])
                business = row['business']
                behaviours = row['behaviours']
                bank = row['bank']
                community = None
                avg_fan_in = random.uniform(row['fan_in_min'], row['fan_in_max'])
                avg_fan_out = random.uniform(row['fan_out_min'], row['fan_out_max'])
                min_amount = random.gauss(row['min_amount'], row['min_amount']/6)
                max_amount = random.gauss(row['max_amount'], row['max_amount']/6)
                avg_amount = random.gauss(row['avg_amount'], row['avg_amount']/6)
                new_beneficiary_ratio = random.gauss(row['balance'], row['balance']/6)
                compromising_ratio = random.uniform(row['compromising_ratio']-0.1, row['compromising_ratio']+0.1)
                role = ROLES['Normal']

                account = Account(acct_id, balance, business, behaviours, bank, community, avg_fan_in, avg_fan_out, min_amount, max_amount, avg_amount,
                                  new_beneficiary_ratio, compromising_ratio, role)

                self.accounts[acct_id] = account
                acct_id += 1

    def load_banks(self, banks_file):
        bank_df = pd.read_csv(banks_file)
        bank_id = 0
        for _, row in bank_df.iterrows():
            self.banks[bank_id] = Bank(bank_id, row['compromising_ratio'], row['launderer_ratio'])
            bank_id += 1

        assert sum([bank.launderer_ratio for _, bank in self.banks.items()]) == 1


    def create_launderers(self, launderer_prob, launderer_dist):
        self.num_accounts = len(self.accounts)
        launderers_num = self.num_accounts * launderer_prob

        bank_launderer_risk_distr = [bank.launderer_ratio for _, bank in self.banks.items()]

        # Each one is a list which contains, for each bank, the number of source/layer/dest nodes related to that bank
        source_dist = bank_launderer_risk_distr * (launderers_num * launderer_dist[0])
        layerer_dist = bank_launderer_risk_distr * (launderers_num * launderer_dist[1])
        destination_dist = bank_launderer_risk_distr * (launderers_num * launderer_dist[2])

        # Create dataframe from accounts. It is done to speed up the computation
        acc_df = pd.DataFrame([acc.to_dataframe_row for _, acc in self.accounts.items()], columns=Account.get_dataframe_columns())

        # The following operations are used to remove the most active accounts. This is done by first extracting the 0.90 quantile from the avg_fan_out
        # distribution adn then removing the account with higher avg_fan_out. Launderers does not do many transactions in order to not be bothered
        quantile_df = acc_df[['avg_fan_out']]
        tx_out_quant = np.quantile(quantile_df, 0.90, axis=0)
        acc_df.drop(acc_df[acc_df['avg_fan_out'] >= tx_out_quant].index, inplace=True)

        # For each bank are extracted sources, layers and destinations
        for bank_id in range(0, len(bank_launderer_risk_distr)):
            bank_users_idxs = set(acc_df[acc_df[[5] == bank_id]][[0]])

            # Set AMLSources for bank_i
            source_ids = random.sample(bank_users_idxs, k=source_dist[bank_id])
            for i in source_ids:
                self.accounts[i].role = ROLES['AMLSource']
                self.aml_sources.append(self.accounts[i].id)

            # Set AMLLayer for bank_i
            bank_users_idxs = bank_users_idxs - set(source_ids)
            layer_ids = random.sample(bank_users_idxs, k=layerer_dist[bank_id])
            for i in layer_ids:
                self.accounts[i].role = ROLES['AMLLayer']
                self.aml_layerer.append(self.accounts[i].id)

            # Set AMLDestination for bank_i
            bank_users_idxs = bank_users_idxs - set(layer_ids)
            destination_ids = random.sample(bank_users_idxs, k=destination_dist[bank_id])
            for i in destination_ids:
                self.accounts[i].role = ROLES['AMLDestination']
                self.aml_destinations.append(self.accounts[i].id)

        self.num_launderers = len(self.aml_sources) + len(self.aml_layerer) + len(self.aml_destinations)
        assert self.num_launderers in range(launderers_num*0.9, launderers_num*1.1)


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
                accounts_num = random.uniform(min_accounts, max_accounts)
                pattern = Pattern(normal_patterns_id, pattern_type, period, amount, scheduling_type=SCHEDULING[scheduling])

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

                pattern = Pattern(aml_patterns_id, pattern_type, period, amount, scheduling_type=SCHEDULING[scheduling])

                accounts_dist = pattern.get_account_distribution(accounts_num)
                # TODO: maybe a FIFO queue is better to distribute the roles?
                sources = random.sample(self.aml_sources, accounts_dist[0])
                pattern.add_source(sources)
                layers = random.sample(self.aml_layerer, accounts_dist[1])
                pattern.add_layerer(layers)
                destinations = random.sample(self.aml_destinations, accounts_dist[2])
                pattern.add_destination(destinations)




                start_time = min(pattern.scheduling_times)
                if start_time not in self.aml_queue.keys():
                    self.aml_queue[start_time] = [pattern]
                else:
                    self.aml_queue[start_time].append(pattern)






