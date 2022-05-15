import csv


class DataWriter:
    def __int__(self, folder: str, account_info_file: str, transaction_file: str, normal_pattern_file: str,
                ml_pattern_file: str):
        folder = "../" + folder
        self.account_info_file = folder + account_info_file
        self.transaction_file = folder + transaction_file
        self.normal_pattern_file = folder + normal_pattern_file
        self.ml_pattern_file = folder + ml_pattern_file

        #self.__init_files()

    def __init_files(self) -> None:
        # Insert header into accounts.txt
        with open(self.account_info_file, 'a') as f:
            writer = csv.writer(f)
            header = ['id', 'business', 'balance', 'balance', 'behaviours', 'nationality', 'bank_id', 'avg_tx_per_step',
                      'min_amount', 'max_amount', 'new_beneficiary_ratio', 'new_neighbours_ratio', 'compromising_ratio',
                      'role']
            writer.writerow(header)

        # Insert header into transactions.txt
        with open(self.transaction_file, 'a') as f:
            writer = csv.writer(f)
            header = ['id', 'src', 'dst', 'amt', 'time', 'type']
            writer.writerow(header)

    def write_accounts_info(self, accounts: list) -> None:
        with open(self.account_info_file, 'a') as f:
            writer = csv.writer(f)
            for account in accounts:
                row = [account.id, account.business, account.balance, account.behaviours, account.nationality,
                       account.bank_id, account.avg_tx_per_step, account.min_amount, account.max_amount,
                       account.new_beneficiary_ratio, account.new_neighbour_ratio, account.compromising_ratio,
                       account.role]
                writer.writerow(row)

    def write_transactions(self, transactions: list) -> None:
        with open(self.transaction_file, 'a') as f:
            writer = csv.writer(f)
            for id, src, dst, amt, time, type in transactions:
                writer.writerow([id, src, dst, amt, time, type])
