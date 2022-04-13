import random

NORMAL_PATTERNS = {'Fan_in': 0, 'Fan_out': 1, 'Cycle': 2, 'Forward': 3, 'Mutual': 4, 'Periodical': 5}
ML_PATTERNS = {'Fan_in_r': 0, 'Fan_out_r': 1, 'Cycle': 2, 'Scatter_Gather': 3, 'Gather_Scatter': 4, 'U': 5, 'Repeated': 6, 'Cash_in': 7, 'Cash_out': 8}
ROLES = {'Normal': 0, 'Mule': 1, 'AMLSource': 2, 'AMLLayer': 3, 'AMLDestination': 4}
SCHEDULING = {'Instant': 0, 'Periodic': 1, 'Random': 2}


class Account:
    def __init__(self, acct_id, balance, business, behaviours, bank, community, avg_fan_in, avg_fan_out, min_amount, max_amount, avg_amount,
                 new_beneficiary_ratio, compromising_ratio, role):
        self.id = acct_id
        self.balance = balance
        self.business = business
        self.behaviours = behaviours
        self.nationality = None
        self.bank = bank
        self.community = community
        self.known_nodes = []
        self.avg_fan_in = avg_fan_in
        self.avg_fan_out = avg_fan_out
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.avg_amount = avg_amount
        self.new_beneficiary_ratio = new_beneficiary_ratio
        self.compromising_ratio = compromising_ratio
        self.role = role

    @classmethod
    def get_dataframe_columns(cls):
        return ['id', 'balance', 'business', 'behaviours', 'nationality', 'bank', 'community', 'known_nodes', 'avg_fan_in', 'avg_fan_out', 'min_amount',
                'max_amount', 'avg_amount', 'new_beneficiary_ratio', 'compromising_ratio', 'role']

    def set_known_nodes(self, known_nodes):
        self.known_nodes = known_nodes

    def add_known_node(self, known_node):
        self.known_nodes.append(known_node)

    def get_beneficiary(self):
        new_beneficiary = random.random() < self.new_beneficiary_ratio
        if new_beneficiary:
            return -1
        return random.choice(self.known_nodes)

    def to_dataframe_row(self):
        return [self.id, self.balance, self.business, self.behaviours, self.nationality, self.bank, self.community, self.known_nodes, self.avg_fan_in,
                self.avg_fan_out, self.min_amount, self.max_amount, self.avg_amount, self.new_beneficiary_ratio, self.compromising_ratio, self.role]


class Bank:
    def __init__(self, bank_id, compromising_ratio, launderer_ratio):
        self.id = bank_id
        self.compromising_ratio = compromising_ratio
        self.launderer_ratio = launderer_ratio


class Pattern:
    def __init__(self, pattern_id, pattern_type, period, amount, sources=None, layers=None, destinations=None, scheduling_type=SCHEDULING['Random']):
        self.id = pattern_id
        self.pattern = pattern_type
        self.scheduling_times = set()
        self.scheduling_type = scheduling_type
        self.period = period
        self.amount = amount
        self.sources = sources if sources is not None else set()
        self.layers = layers if layers is not None else set()
        self.destinations = destinations if destinations is not None else set()
        self.transactions = dict()

    def get_account_distribution(self, num_accounts):
        in_out_dict = {'0': [num_accounts-1, 0, 1], '1': [1, 0, num_accounts-1], '2': [0, num_accounts-1, 0],
                       '3': [1, num_accounts-1, 1], '4': [(num_accounts-1)/2, 1, (num_accounts-1)/2],
                       '5': [1, num_accounts-1, 0], '6': [1, 0, 1], '7': 7, '8': 8}

        return in_out_dict[self.pattern]

    def add_source(self, acct_id):
        self.sources.add(acct_id)

    def add_layerer(self, acct_id):
        self.layers.add(acct_id)

    def add_destination(self, acct_id):
        self.destinations.add(acct_id)

    def schedule(self, max_time):
        start_time = random.uniform(0, max_time-self.period-1)
        end_time = start_time + self.period

        ML_PATTERNS = {'Fan_in_r': 0, 'Fan_out_r': 1, 'Cycle': 2, 'Scatter_Gather': 3, 'Gather_Scatter': 4, 'U': 5, 'Repeated': 6, 'Cash_in': 7,
                       'Cash_out': 8}
        in_out_dict = {'0': len(self.sources), '1': len(self.destinations), '2': len(self.layers), '3': 2*len(self.layers),
                       '4': len(self.sources) + len(self.destinations), '5': (len(self.layers)+1)*2 - 2,
                       '6': random.uniform(8, 12), '7': 1, '8': 1}

    def update(self, time):
        self.scheduling_times = self.scheduling_type - time
        del self.transactions[time]





