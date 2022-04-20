import random
import numpy as np
import pandas as pd

from src._constants import *

# Accounts constants dictionaries
BUSINESS_TYPE = {'Retail': 0, 'Corporate': 1}


class Population:
    def __init__(self):
        self.accounts = dict()
        self.bank_to_acc = dict()
        self.compromising_accounts_ratio = list()
        self.compromised = set()
        self.aml_sources = []
        self.aml_layerer = []
        self.aml_destinations = []

    def get_accounts(self):
        return self.accounts

    def get_accounts_ids(self):
        return self.accounts.keys()

    def get_accounts_num(self):
        return len(self.accounts)

    def get_accounts_from_bank(self, bank_id):
        return self.bank_to_acc[bank_id]

    def add_account(self, account):
        account_id = account.id
        self.accounts[account_id] = account

        bank_id = account.bank_id
        if bank_id in self.bank_to_acc:
            self.bank_to_acc[bank_id].append(account_id)
        else:
            self.bank_to_acc[bank_id] = list(account_id)

        self.compromising_accounts_ratio.append(account.compromising_ratio)

    def add_compromised(self, account_id):
        self.compromised.add(account_id)

    def create_launderers(self, launderer_prob, banks_laundering_ratio, launderer_dist):
        num_accounts = len(self.accounts)
        num_banks = len(banks_laundering_ratio)
        launderers_num = int(num_accounts * launderer_prob)

        # Each one is a list which contains, for each bank, the number of source/layer/dest nodes related to that bank
        source_dist = banks_laundering_ratio * (launderers_num * launderer_dist['AMLSources'])
        layerer_dist = banks_laundering_ratio * (launderers_num * launderer_dist['AMLLayers'])
        destination_dist = banks_laundering_ratio * (launderers_num * launderer_dist['AMLDestinations'])

        # Create dataframe from accounts. It is done to speed up the computation
        acc_df = pd.DataFrame([acc.to_dataframe_row for _, acc in self.accounts.items()], columns=Account.get_dataframe_columns())

        # The following operations are used to remove the most active accounts. This is done by first extracting the 0.90 quantile from the avg_fan_out
        # distribution adn then removing the account with higher avg_fan_out. Launderers does not do many transactions in order to not be bothered
        quantile_df = acc_df[['avg_fan_out']]
        tx_out_quant = np.quantile(quantile_df, 0.90, axis=0)
        acc_df.drop(acc_df[acc_df['avg_fan_out'] >= tx_out_quant].index, inplace=True)

        # For each bank are extracted sources, layers and destinations
        for bank_id in range(0, num_banks):
            bank_users_idxs = set(acc_df[acc_df[[5] == bank_id]][[0]])

            # Set AMLSources for bank_i
            source_ids = random.sample(bank_users_idxs, k=source_dist[bank_id])
            for i in source_ids:
                self.accounts[i].role = AML_SOURCE
                self.aml_sources.append(self.accounts[i].id)

            # Set AMLLayer for bank_i
            bank_users_idxs = bank_users_idxs - set(source_ids)
            layer_ids = random.sample(bank_users_idxs, k=layerer_dist[bank_id])
            for i in layer_ids:
                self.accounts[i].role = AML_LAYER
                self.aml_layerer.append(self.accounts[i].id)

            # Set AMLDestination for bank_i
            bank_users_idxs = bank_users_idxs - set(layer_ids)
            destination_ids = random.sample(bank_users_idxs, k=destination_dist[bank_id])
            for i in destination_ids:
                self.accounts[i].role = AML_DESTINATION
                self.aml_destinations.append(self.accounts[i].id)

        num_launderers = len(self.aml_sources) + len(self.aml_layerer) + len(self.aml_destinations)
        assert num_launderers in range(int(launderers_num * 0.9), int(launderers_num * 1.1))

        return num_launderers

    def get_compromised(self, num_compromised):
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


class Account:
    def __init__(self, acct_id, balance, business, behaviours, bank_id, community, avg_fan_in, avg_fan_out, min_amount,
                 max_amount, avg_amount, compromising_ratio, role):
        self.id = acct_id
        self.balance = balance
        self.business = BUSINESS_TYPE[business]
        self.behaviours = behaviours
        self.nationality = None
        self.bank_id = bank_id
        self.community = community
        self.known_nodes = set()
        self.avg_fan_in = avg_fan_in
        self.avg_fan_out = avg_fan_out
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.avg_amount = avg_amount
        self.new_beneficiary_ratio = random.gauss(DEFAULT_NEW_BENE_RATIO[self.business], 0.002)
        self.new_neighbour_ratio = random.gauss(DEFAULT_NEW_NEIGHBOUR_RATIO[self.business], 0.05)
        self.compromising_ratio = compromising_ratio
        self.role = role

    @classmethod
    def get_dataframe_columns(cls):
        return ['id', 'balance', 'business', 'behaviours', 'nationality', 'bank', 'community', 'known_nodes', 'avg_fan_in', 'avg_fan_out', 'min_amount',
                'max_amount', 'avg_amount', 'new_beneficiary_ratio', 'compromising_ratio', 'role']

    def set_known_nodes(self, known_nodes):
        self.known_nodes = known_nodes

    def add_known_node(self, known_node):
        self.known_nodes.add(known_node)

    def get_transaction(self):
        new_beneficiary = random.random() < self.new_beneficiary_ratio
        if new_beneficiary:
            beneficiary = -1
        else:
            beneficiary = random.choice(list(self.known_nodes))

        amount = random.uniform(self.min_amount, self.max_amount)

        return beneficiary, -amount

    def update(self, beneficiary, amount):
        if self.balance + amount < 0:
            return False
        self.balance += amount

        if (beneficiary is not None) and (beneficiary not in self.known_nodes):
            if random.random() > self.new_neighbour_ratio:
                self.known_nodes.add(beneficiary)

        return True

    def to_dataframe_row(self):
        return [self.id, self.balance, self.business, self.behaviours, self.nationality, self.bank_id, self.community, self.known_nodes, self.avg_fan_in,
                self.avg_fan_out, self.min_amount, self.max_amount, self.avg_amount, self.new_beneficiary_ratio, self.compromising_ratio, self.role]


class Bank:
    def __init__(self, bank_id, compromising_ratio, launderer_ratio):
        self.id = bank_id
        self.compromising_ratio = compromising_ratio
        self.launderer_ratio = launderer_ratio


class Community:
    def __init__(self, community_id, members):
        self.id = community_id
        self.members = set(members)
        self.bridges = random.sample(self.members, k=max(1, int(len(self.members)*0.05)))

    def add_member(self, node_id):
        self.members.add(node_id)







