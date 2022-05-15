import random
import yaml

import pandas as pd

from src._constants import ACCOUNT
from src._variables import SIMULATION
from src.model.population import Population, Account, Bank
from src.model.community import Community
from src.model.datawriter import DataWriter
from src.model.simulation import Simulation
from src.model.pattern import create_pattern
from src.utils import scheduling_string_to_const, pattern_string_to_const


class AMLDataGen:
    def __init__(self, _conf_file):
        # Open config file
        with open(_conf_file, "r") as rf:
            try:
                self.config = yaml.safe_load(rf)
            except yaml.YAMLError as exc:
                print(exc)
                exit()

        # Entities
        self.banks = dict()
        self.population = Population()
        self.community = Community()

        # Patterns
        self.normal_patterns = list()
        self.ml_patterns = list()

        # Simulator Data
        self.__simulation = None
        self.end_time = SIMULATION.END_TIME

        # Output files
        output_files = self.config['Output_files']
        folder = output_files['output_folder'] + self.config['Simulation_name'] + '/'
        accounts_out_file = output_files['account_file']
        transaction_out_file = output_files['transaction_file']

        self.writer = DataWriter()

    # LOADERS
    # ------------------------------------------

    def load_all(self):
        input_files = self.config['Input_files']
        input_folder = "../" + input_files['input_folder']

        bank_file = input_folder + input_files['bank_file']
        self.load_banks(bank_file)

        accounts_file = input_folder + input_files['account_file']
        self.load_accounts(accounts_file)

        normal_pattern_file = input_folder + input_files['normal_pattern_file']
        self.load_normal_patterns(normal_pattern_file)

        aml_pattern_file = input_folder + input_files['ml_pattern_file']
        self.load_ml_patterns(aml_pattern_file)

    def load_banks(self, banks_file):
        bank_df = pd.read_csv(banks_file)
        bank_id = 0
        for _, row in bank_df.iterrows():
            self.banks[bank_id] = Bank(bank_id, row['bank_name'], row['compromising_ratio'], row['launderer_ratio'])
            bank_id += 1

        assert sum([bank.launderer_ratio for _, bank in self.banks.items()]) == 1

    def load_accounts(self, accounts_file):
        acc_df = pd.read_csv(accounts_file)
        acct_id = 0
        for _, row in acc_df.iterrows():
            for _ in range(0, row['count']):
                balance = random.uniform(row['balance_min'], row['balance_max'])
                balance_limit_percentage = row['balance_limit_percentage']
                business = row['business']
                behaviours = row['behaviours']
                print("bh:" + str(behaviours))
                bank_id = row['bank_id'] if row['bank_id'] != 'rnd' else random.choice(range(0, len(self.banks)))
                print("bank: " + str(bank_id))
                community = None
                avg_tx_per_step = row['avg_tx_per_step']
                min_amount = random.gauss(row['min_tx_amount'], row['min_tx_amount']/6)
                max_amount = random.gauss(row['max_tx_amount'], row['max_tx_amount']/6)
                compromising_ratio = random.uniform(row['compromising_ratio']-0.1, row['compromising_ratio']+0.1)
                role = ACCOUNT.NORMAL

                account = Account(acct_id, balance, balance_limit_percentage, business, behaviours, bank_id,
                                  avg_tx_per_step, min_amount, max_amount, compromising_ratio, role)

                self.population.add_account(account)
                self.community.add_node(account)
                acct_id += 1

    def load_normal_patterns(self, normal_patterns_file):
        normal_patterns_df = pd.read_csv(normal_patterns_file)
        normal_patterns_id = 0
        for _, row in normal_patterns_df.iterrows():
            pattern_type = pattern_string_to_const(row['pattern_type'])
            period = row['period']
            min_amount = row['min_amount']
            max_amount = row['max_amount']
            min_accounts = row['min_accounts']
            max_accounts = row['max_accounts']
            scheduling = scheduling_string_to_const(row['scheduling'])
            for _ in row['count']:
                amount = random.uniform(min_amount, max_amount)
                accounts_num = random.randint(min_accounts, max_accounts)
                pattern = create_pattern(normal_patterns_id, pattern_type, accounts_num, period, amount, False,
                                         scheduling_type=scheduling)
                self.normal_patterns.append(pattern)
                normal_patterns_id += 1

    def load_ml_patterns(self, aml_patterns_file):
        ml_patterns_df = pd.read_csv(aml_patterns_file)
        ml_patterns_id = 0
        for _, row in ml_patterns_df.iterrows():
            pattern_type = pattern_string_to_const(row['pattern_type'])
            period = row['period']
            min_amount = row['min_amount']
            max_amount = row['max_amount']
            min_accounts = row['min_accounts']
            max_accounts = row['max_accounts']
            scheduling = scheduling_string_to_const(row['scheduling'])
            for _ in row['count']:
                amount = random.uniform(min_amount, max_amount)
                accounts_num = random.randint(min_accounts, max_accounts)
                pattern = create_pattern(ml_patterns_id, pattern_type, accounts_num, period, amount, True,
                                         scheduling_type=scheduling)
                self.normal_patterns.append(pattern)
                ml_patterns_id += 1

    # SIMULATION
    # ------------------------------------------

    def create_simulation(self):
        self.__simulation = Simulation(self.population, self.community, start_time=0, end_time=self.end_time)
        self.__simulation.load_normal_patterns(self.normal_patterns)
        self.__simulation.load_ml_patterns(self.ml_patterns)

    def run_simulation(self):
        self.__simulation.start()

    def simulation_step(self):
        self.__simulation.step(self.__simulation.time)


if __name__ == "__main__":
    conf_file = "../Inputs/_config.yaml"

    aml_data_gen = AMLDataGen(conf_file)
    aml_data_gen.load_all()




