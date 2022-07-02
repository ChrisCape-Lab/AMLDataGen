import os
import sys
import shutil
import pandas as pd
import logging
import random
import yaml

import src._constants as _c
import src._variables as _v
from src.model.population import Population, Account, Bank
from src.model.datawriter import DataWriter
from src.model.simulation import Simulation
from src.model.pattern import create_pattern


SINGLE = 0
SERIES = 1


# RUN
# ------------------------------------------

def run(config_file_name):
    with open(config_file_name, "r") as rf:
        try:
            config = yaml.safe_load(rf)
            out_path = "../" + config['Output_files']['output_folder'] + config['Simulation_name'] + '/'
            sim_name = config['Simulation_name']
        except yaml.YAMLError as exc:
            print(exc)
            exit()

    print("Running " + sim_name)
    _v.load_variables(config['configuration_parameters'])

    logging_path = '../Logs/' + sim_name + '.log'
    _v.logger = my_custom_logger(logging_path)
    #logging.basicConfig(filename=logging_path, filemode='w', format='[%(levelname)s] %(message)s', level=logging.INFO)

    aml_data_gen = AMLDataGen(conf_file)
    aml_data_gen.load_all()

    aml_data_gen.create_simulation()

    aml_data_gen.run_simulation()
    del aml_data_gen

    shutil.copy(logging_path, out_path)


# UTILS
# ------------------------------------------


def my_custom_logger(log_file_path, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(log_file_path)
    logger.setLevel(level)
    log_format = logging.Formatter('[%(levelname)s] %(message)s')

    # Creating and adding the file handler
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger


def scheduling_string_to_const(scheduling_str: str) -> int:
    if scheduling_str == 'Random':
        return _c.SCHEDULING.RANDOM
    elif scheduling_str == 'Periodic':
        return _c.SCHEDULING.PERIODIC
    elif scheduling_str == 'Instant':
        return _c.SCHEDULING.INSTANT
    else:
        exception_string = "The scheduling '" + scheduling_str + "' is not implemented, sorry!"
        raise NotImplementedError


def pattern_string_to_const(pattern_str: str) -> int:
    if pattern_str.lower() == 'random':
        return _c.PTRN_TYPE.RANDOM_P
    elif pattern_str.lower() == 'fan_in':
        return _c.PTRN_TYPE.FAN_IN
    elif pattern_str.lower() == 'fan_out':
        return _c.PTRN_TYPE.FAN_OUT
    elif pattern_str.lower() == 'cycle':
        return _c.PTRN_TYPE.CYCLE
    elif pattern_str.lower() == 'scatter_gather':
        return _c.PTRN_TYPE.SCATTER_GATHER
    elif pattern_str.lower() == 'gather_scatter':
        return _c.PTRN_TYPE.GATHER_SCATTER
    elif pattern_str.lower() == 'u_pattern':
        return _c.PTRN_TYPE.U
    elif pattern_str.lower() == 'repeated':
        return _c.PTRN_TYPE.REPEATED
    elif pattern_str.lower() == 'bipartite':
        return _c.PTRN_TYPE.BIPARTITE
    else:
        exception_string = "The pattern '" + pattern_str + "' is not implemented, sorry!"
        raise NotImplementedError(exception_string)


# AMLDATAGEN CLASS
# ------------------------------------------


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

        # Patterns
        self.normal_patterns = list()
        self.ml_patterns = list()

        # Simulator Data
        self.__simulation = None
        self.end_time = _v.SIM.DEF_END_TIME

        # Output files
        output_files = self.config['Output_files']
        folder = output_files['output_folder'] + self.config['Simulation_name'] + '/'
        accounts_out_file = output_files['account_file']
        transaction_out_file = output_files['transaction_file']
        degree_out_file = output_files['degree_file']
        normal_patterns_file = output_files['normal_patterns_file']
        ml_patterns_file = output_files['ml_patterns_file']

        self.data_writer = DataWriter(folder, accounts_out_file, transaction_out_file, degree_out_file, normal_patterns_file,
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
        print("Init: banks loaded correctly")

    def load_accounts(self, accounts_file):
        acc_df = pd.read_csv(accounts_file)
        acct_id = 0
        for _, row in acc_df.iterrows():
            for _ in range(0, row['count']):
                balance = random.uniform(row['balance_min'], row['balance_max'])
                balance_limit_percentage = row['balance_limit_percentage']
                business = row['business']
                behaviours = row['behaviours']
                bank_id = row['bank_id'] if row['bank_id'] != 'rnd' else random.choice(range(0, len(self.population.get_bank_ids())))
                community = None
                avg_tx_per_step = row['avg_tx_per_step']
                min_amount = random.gauss(row['min_tx_amount'], row['min_tx_amount']/6)
                max_amount = random.gauss(row['max_tx_amount'], row['max_tx_amount']/6)
                compromising_ratio = random.uniform(row['compromising_ratio']-0.05, row['compromising_ratio']+0.05)
                role = _c.ACCTS_ROLES.NORMAL

                assert balance > 0 and min_amount > 0 and max_amount > 0
                account = Account(acct_id, balance, balance_limit_percentage, business, behaviours, bank_id,
                                  avg_tx_per_step, min_amount, max_amount, compromising_ratio, role)

                self.population.add_account(account)
                acct_id += 1
        out = "Init: accounts loaded correctly"
        print(out)
        _v.logger.info(out)

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

                assert accounts_num > 0 and period > 0 and amount > 0
                pattern = create_pattern(normal_patterns_id, pattern_type, accounts_num, period, amount, False,
                                         scheduling_type=scheduling)
                self.normal_patterns.append(pattern)
                normal_patterns_id += 1
        out = "Init: normal patterns loaded correctly"
        print(out)
        _v.logger.info(out)

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

                assert accounts_num > 0 and period > 0 and amount > 0
                pattern = create_pattern(ml_patterns_id, pattern_type, accounts_num, period, amount, True,
                                         scheduling_type=scheduling)
                self.normal_patterns.append(pattern)
                ml_patterns_id += 1
        out = "Init: laundering patterns loaded correctly"
        print(out)
        _v.logger.info(out)

    # SIMULATION
    # ------------------------------------------

    def create_simulation(self):
        self.__simulation = Simulation(self.population, self.data_writer, start_time=0, end_time=self.end_time)
        self.__simulation.load_normal_patterns(self.normal_patterns)
        self.__simulation.load_ml_patterns(self.ml_patterns)
        out = "Init: simulation created correctly"
        print(out)
        _v.logger.info(out)

    def run_simulation(self):
        out = "sim: Starting simulation"
        print(out)
        _v.logger.info(out)
        self.__simulation.setup(_v.SIM.DEF_ALLOW_RANDOM_TXS)
        self.__simulation.run()

    def simulation_step(self):
        self.__simulation.step(self.__simulation.time)


if __name__ == "__main__":
    input_directory = "../Inputs/"
    conf_file = input_directory + "_Sim_test/_config.yaml"

    mode = SERIES
    if mode == SINGLE:
        run(conf_file)
    elif mode == SERIES:
        for experiment in os.listdir(input_directory):
            experiment_dir = os.path.join(input_directory, experiment)
            if os.path.isfile(experiment_dir) or experiment == "_Sim_test" or experiment == "_Other_sims" or experiment == '_tests' or experiment == 'tmp':
                continue
            conf_file = input_directory + str(experiment) + "/_config.yaml"
            run(conf_file)
