import os
import csv


class DataWriter:
    def __init__(self, folder: str, account_info_file: str, transaction_file: str, degree_file: str, normal_pattern_file: str,
                 ml_pattern_file: str):
        folder = "../" + folder
        self.account_info_file = folder + account_info_file
        self.degree_file = folder + degree_file
        self.transaction_file = folder + transaction_file
        self.normal_pattern_file = folder + normal_pattern_file
        self.ml_pattern_file = folder + ml_pattern_file

        self.__init_files(folder)

    def __init_files(self, folder) -> None:
        # Check if simulation output folder exists, otherwise creates it
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Insert header into accounts.csv
        with open(self.account_info_file, 'w', newline="") as f:
            writer = csv.writer(f)
            header = ['id', 'business', 'init_balance', 'behaviours', 'nationality', 'bank_id', 'avg_tx_per_step',
                      'min_amount', 'max_amount', 'new_beneficiary_ratio', 'new_neighbours_ratio', 'compromising_ratio',
                      'role']
            writer.writerow(header)

        # Insert header into degree.csv
        with open(self.degree_file, 'w', newline="") as f:
            writer = csv.writer(f)
            header = ['Count', 'In-degree', 'Out-degree']
            writer.writerow(header)

        # Insert header into transactions.csv
        with open(self.transaction_file, 'w', newline="") as f:
            writer = csv.writer(f)
            header = ['id', 'src', 'dst', 'amt', 'time', 'type', 'is_aml']
            writer.writerow(header)

    def write_accounts_info(self, accounts: list) -> None:
        with open(self.account_info_file, 'a', newline="") as f:
            writer = csv.writer(f)
            for account in accounts:
                row = [account.id, account.business, account.balance, account.behaviours, account.nationality,
                       account.bank_id, account.avg_tx_per_step, account.min_amount, account.max_amount,
                       account.new_beneficiary_ratio, account.new_neighbour_ratio, account.compromising_ratio,
                       account.role]
                writer.writerow(row)

    def write_degrees(self, degrees: list):
        with open(self.degree_file, 'a', newline="") as f:
            writer = csv.writer(f)
            for degree in degrees:
                writer.writerow(degree)

    def write_transactions(self, transactions: list) -> None:
        with open(self.transaction_file, 'a', newline="") as f:
            writer = csv.writer(f)
            for id, src, dst, amt, time, type, is_aml in transactions:
                writer.writerow([id, src, dst, amt, time, type, is_aml])
