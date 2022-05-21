import logging
import os
import random
import yaml

import pandas as pd

import src._constants as _c
import src._variables as _v
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
        self.population = Population()
        self.community = Community()

        # Patterns
        self.normal_patterns = list()
        self.ml_patterns = list()

        # Simulator Data
        self.__simulation = None
        self.end_time = _v.SIMULATION.DEF_END_TIME

        # Output files
        output_files = self.config['Output_files']
        folder = output_files['output_folder'] + self.config['Simulation_name'] + '/'
        accounts_out_file = output_files['account_file']
        transaction_out_file = output_files['transaction_file']
        normal_patterns_file = output_files['normal_patterns_file']
        ml_patterns_file = output_files['ml_patterns_file']

        self.data_writer = DataWriter(folder, accounts_out_file, transaction_out_file, normal_patterns_file,
                                      ml_patterns_file)

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
            self.population.add_bank(Bank(bank_id, row['bank_name'], row['compromising_ratio'], row['launderer_ratio']))
            bank_id += 1
        print("Init: Banks loaded correctly")

    def load_accounts(self, accounts_file):
        acc_df = pd.read_csv(accounts_file)
        acct_id = 0
        for _, row in acc_df.iterrows():
            for _ in range(0, row['count']):
                balance = random.uniform(row['balance_min'], row['balance_max'])
                balance_limit_percentage = row['balance_limit_percentage']
                business = row['business']
                behaviours = row['behaviours']
                bank_id = row['bank_id'] if row['bank_id'] != 'rnd' else random.choice(range(0, self.population.get_bank_nums()))
                community = None
                avg_tx_per_step = row['avg_tx_per_step']
                min_amount = random.gauss(row['min_tx_amount'], row['min_tx_amount']/6)
                max_amount = random.gauss(row['max_tx_amount'], row['max_tx_amount']/6)
                compromising_ratio = random.uniform(row['compromising_ratio']-0.1, row['compromising_ratio']+0.1)
                role = _c.ACCOUNT.NORMAL

                account = Account(acct_id, balance, balance_limit_percentage, business, behaviours, bank_id,
                                  avg_tx_per_step, min_amount, max_amount, compromising_ratio, role)

                self.population.add_account(account)
                self.community.add_node(account)
                acct_id += 1
        out = "Init: accounts loaded correctly"
        print(out)
        logging.info(out)

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
            for _ in range(0, row['count']):
                amount = random.uniform(min_amount, max_amount)
                accounts_num = random.randint(min_accounts, max_accounts)
                pattern = create_pattern(normal_patterns_id, pattern_type, accounts_num, period, amount, False,
                                         scheduling_type=scheduling)
                self.normal_patterns.append(pattern)
                normal_patterns_id += 1
        out = "Init: normal patterns loaded correctly"
        print(out)
        logging.info(out)

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
            for _ in range(0, row['count']):
                amount = random.uniform(min_amount, max_amount)
                accounts_num = random.randint(min_accounts, max_accounts)
                pattern = create_pattern(ml_patterns_id, pattern_type, accounts_num, period, amount, True,
                                         scheduling_type=scheduling)
                self.normal_patterns.append(pattern)
                ml_patterns_id += 1
        out = "Init: laundering patterns loaded correctly"
        print(out)
        logging.info(out)

    # SIMULATION
    # ------------------------------------------

    def create_simulation(self):
        self.__simulation = Simulation(self.population, self.community, self.data_writer, start_time=0,
                                       end_time=self.end_time)
        self.__simulation.load_normal_patterns(self.normal_patterns)
        self.__simulation.load_ml_patterns(self.ml_patterns)
        out = "Sim: simulation created correctly"
        print(out)
        logging.info(out)

    def run_simulation(self):
        out = "Sim: Starting simulation"
        print(out)
        logging.info(out)
        self.__simulation.setup(_v.SIMULATION.DEF_ALLOW_RANDOM_TXS)
        self.__simulation.run()

    def simulation_step(self):
        self.__simulation.step(self.__simulation.time)


if __name__ == "__main__":
    conf_file = "../Inputs/_config.yaml"
    with open(conf_file, "r") as rf:
        try:
            config = yaml.safe_load(rf)
            sim_name = config['Simulation_name']
        except yaml.YAMLError as exc:
            print(exc)
            exit()
    logging_path = '../Logs/' + sim_name + '.log'
    logging.basicConfig(filename=logging_path, filemode='w', format='[%(levelname)s] %(message)s', level=logging.INFO)

    aml_data_gen = AMLDataGen(conf_file)
    aml_data_gen.load_all()

    aml_data_gen.create_simulation()

    aml_data_gen.run_simulation()



