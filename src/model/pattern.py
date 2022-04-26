import random

import networkx as nx

from src._constants import *
from src._variables import *
from src.utils import random_chunk, add_to_dict_of_list, addn_to_dict_of_list


def get_rounded(value):
    if value > 0:
        amount = round(value, 2)
        digits = len(str(amount))
        most_significant_nums = str(amount)[0:int(digits/2)]
        rounded_amt = str(most_significant_nums).ljust(digits, '0')

        return int(rounded_amt)
    else:
        print("Negative Amount!")


def get_under_threshold(value):
    return get_rounded(value) - 1


class Pattern:
    def __init__(self, pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio=.0, under_threshold_ratio=.0, scheduling_type=RANDOM):
        self.id = pattern_id
        self.pattern = pattern_type
        self.num_acconts = num_accounts
        self.structure = nx.DiGraph()
        self.accounts = dict()
        self.scheduling_type = scheduling_type
        self.period = period
        self.amount = amount
        self.rounded_ratio = rounded_ratio
        self.under_threshold_ratio = under_threshold_ratio
        self.transactions = dict()
        self.is_aml = is_aml

    # SETTERS
    # ------------------------------------------

    def add_source(self, source):
        add_to_dict_of_list(self.accounts, PATTERN.SOURCE, source)

    def add_sources(self, sources):
        add_to_dict_of_list(self.accounts, PATTERN.SOURCE, sources)

    def add_layerer(self, layerer):
        add_to_dict_of_list(self.accounts, PATTERN.LAYER, layerer)

    def add_layerers(self, layerers):
        add_to_dict_of_list(self.accounts, PATTERN.LAYER, layerers)

    def add_destination(self, destination):
        add_to_dict_of_list(self.accounts, PATTERN.DESTINATION, destination)

    def add_destinations(self, destinations):
        add_to_dict_of_list(self.accounts, PATTERN.DESTINATION, destinations)

    # PRIVATE
    # ------------------------------------------

    def _get_amount(self, value):
        weights = [1 - (self.rounded_ratio + self.under_threshold_ratio), self.rounded_ratio, self.under_threshold_ratio]
        case = random.choices([NORMAL_AMT, ROUNDED, UNDER_THRESH], weights=weights, k=1)
        if case == NORMAL_AMT:
            return round(value, 2)
        elif case == ROUNDED:
            return get_rounded(value)
        elif case == UNDER_THRESH:
            return get_under_threshold(value)
        else:
            raise NotImplementedError

    # REQUIREMENTS
    # ------------------------------------------

    def get_sources_requirements(self):
        num_sources = 0
        requirements = dict()
        for node_id, n_attr in self.structure.nodes(data=True):
            if n_attr['role'] != 's':
                continue

            fan_out = len(self.structure.successors(node_id))
            amount = 0
            for _, e_attr in self.structure[node_id]:
                amount += e_attr['weight']

            requirement_1 = (GENERAL.FAN_OUT, fan_out)
            requirement_2 = (GENERAL.BALANCE, amount)

            requirements[num_sources] = [requirement_1, requirement_2]
            num_sources += 1

        assert len(requirements) == num_sources

        return num_sources, requirements

    def get_layerer_requirements(self):
        num_layerers = 0
        requirements = dict()
        for node_id, n_attr in self.structure.nodes(data=True):
            if n_attr['role'] != 'l':
                continue

            fan_out = len(self.structure.successors(node_id))
            fan_in = len(self.structure.predecessors(node_id))

            amount = 0
            for _, e_attr in self.structure[node_id]:
                amount += e_attr['weight']
            for pre_id in self.structure.predecessors(node_id):
                amount -= G[pre_id][node_id]['weight']

            requirement_1 = (GENERAL.FAN_OUT, fan_out)
            requirement_2 = (GENERAL.FAN_IN, fan_in)
            requirement_3 = (GENERAL.BALANCE, amount)

            requirements[num_layerers] = [requirement_1, requirement_2, requirement_3]
            num_layerers += 1

        assert len(requirements) == num_layerers

        return num_layerers, requirements

    def get_destinations_requirements(self):
        num_destinations = 0
        requirements = dict()
        for node_id, n_attr in self.structure.nodes(data=True):
            if n_attr['role'] != 'd':
                continue

            fan_in = len(self.structure.predecessors(node_id))
            requirement = (GENERAL.FAN_IN, fan_in)

            requirements[num_destinations] = [requirement]
            num_destinations += 1

        assert len(requirements) == num_destinations

        return num_destinations, requirements

    # BEHAVIOUR
    # ------------------------------------------

    def create_structure(self):
        raise NotImplementedError

    def schedule(self):
        raise NotImplementedError


class FanInPattern(Pattern):
    def __init__(self, pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio=.0, under_threshold_ratio=.0, scheduling_type=RANDOM):
        super().__init__(pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio, under_threshold_ratio, scheduling_type)

    # BEHAVIOUR
    # ------------------------------------------

    def create_structure(self):
        destination_node_id = self.num_acconts - 1
        self.structure.add_node(destination_node_id, role='d')

        num_sources = self.num_acconts - 1
        single_amount = self._get_amount(self.amount / num_sources)
        for i in range(0, num_sources):
            self.structure.add_node(i, role='s')
            self.structure.add_edge(i, destination_node_id, weight=single_amount)

    def schedule(self):
        raise NotImplementedError


class FanOutPattern(Pattern):
    def __init__(self, pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio=.0, under_threshold_ratio=.0,
                 scheduling_type=RANDOM):
        super().__init__(pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio, under_threshold_ratio, scheduling_type)

    # BEHAVIOUR
    # ------------------------------------------

    def create_structure(self):
        source_node_id = 0
        self.structure.add_node(source_node_id, role='s')

        num_sources = self.num_acconts - 1
        single_amount = self._get_amount(self.amount / num_sources)
        for i in range(1, num_sources+1):
            self.structure.add_node(i, role='d')
            self.structure.add_edge(source_node_id, i, weight=single_amount)

    def schedule(self):
        raise NotImplementedError


    def get_account_distribution(self, num_accounts):
        # For Scatter-Gather, determine the number of sources in order to set the number of destinations
        src = random.choice(range(2, num_accounts-1))

        # For Bipartite, determine the number of layers and the sources/destinations distribution
        num_sources = int(random.uniform(BIPARTITE_MIN_SOURCES, BIPARTITE_MAX_SOURCES) * num_accounts)
        num_destinations = int(random.uniform(BIPARTITE_MIN_DESTINATIONS, BIPARTITE_MAX_DESTINATIONS) * num_accounts)

        in_out_dict = {
            FAN_IN: [num_accounts-1, 0, 1],
            FAN_OUT: [1, 0, num_accounts-1],
            CYCLE: [0, num_accounts-1, 0],
            SCATTER_GATHER: [1, num_accounts-1, 1],
            GATHER_SCATTER: [src, 1, num_accounts-src],
            U: [1, num_accounts-1, 0],
            REPEATED: [1, 0, 1],
            BIPARTITE: [num_sources, num_accounts - (num_sources+num_destinations), num_destinations],
            CASH_IN: [1, 0, 0],
            CASH_OUT: [0, 0, 1]
        }

        return in_out_dict[self.pattern]

    def get_num_txs_by_type(self, num_sources=0, num_layerer=0, num_destinations=0):
        num_txs_by_type = {
            FAN_IN: num_sources,
            FAN_OUT: num_destinations,
            CYCLE: num_layerer,
            SCATTER_GATHER: 2 * num_layerer,
            GATHER_SCATTER: num_sources + num_destinations,
            U: (num_layerer + 1) * 2 - 2,
            REPEATED: random.uniform(REPEATED_MIN, REPEATED_MAX),
            BIPARTITE: num_sources * num_layerer * num_destinations,
            CASH_IN: random.uniform(CASH_TX_MIN, CASH_TX_MAX),
            CASH_OUT: random.uniform(CASH_TX_MIN, CASH_TX_MAX)
        }

        return num_txs_by_type[self.pattern]

    def get_transactions(self):
        return self.transactions

    def _schedule(self, num_tx, start_time, end_time):
        # INSTANT SCHEDULING: each tx is executed in the same day as others
        if self.scheduling_type == INSTANTS:
            time = random.randint(start_time, end_time)
            self.scheduling_times = [time] * num_tx
        # PERIODIC SCHEDULING: each tx on a different day if possible, otherwise random sample some repeated days
        elif self.scheduling_type == PERIODIC:
            if num_tx > self.period:
                self.scheduling_times = list(range(start_time, end_time))
                exceeding = num_tx - (end_time - start_time)
                self.scheduling_times = sorted(self.scheduling_times + sorted(random.sample(range(start_time, end_time), k=exceeding)))
            else:
                self.scheduling_times = sorted(list(range(start_time, start_time + num_tx)))
        # RANDOM SCHEDULING: just random sample some days in the interval
        else:
            self.scheduling_times = sorted(random.sample(range(start_time, end_time), k=num_tx))

        for t in self.scheduling_times:
            self.transactions[t] = []

    def schedule_tx(self):
        raise NotImplementedError

    def _calculate_amount(self, value):
        weights = [1 - (self.rounded_ratio + self.under_threshold_ratio), self.rounded_ratio, self.under_threshold_ratio]
        case = random.choices([NORMAL_AMT, ROUNDED, UNDER_THRESH], weights=weights, k=1)
        if case == NORMAL_AMT:
            return round(value, 2)
        elif case == ROUNDED:
            return get_rounded(value)
        elif case == UNDER_THRESH:
            return get_under_threshold(value)
        else:
            raise NotImplementedError

    def update(self, time):
        self.scheduling_times = self.scheduling_type - time
        del self.transactions[time]


class GeneralPattern(Pattern):
    def __init__(self, pattern_id, pattern_type, period, amount, is_aml, accounts, rounded_ratio, under_threshold_ratio, scheduling_type=RANDOM):
        super(GeneralPattern, self).__init__(pattern_id, pattern_type, period, amount, is_aml,
                                             rounded_ratio=rounded_ratio, under_threshold_ratio=under_threshold_ratio, scheduling_type=scheduling_type)

        self.accounts = accounts

    def add_accounts(self, acct_id):
        self.accounts.add(acct_id)

    def schedule(self, max_time):
        start_time = random.randint(0, max_time-self.period-1)
        end_time = start_time + self.period

        distrib = self.get_account_distribution(len(self.accounts))
        num_tx = self.get_num_txs_by_type(distrib[0], distrib[1], distrib[2])

        self._schedule(num_tx, start_time, end_time)

    def schedule_tx(self):
        if self.pattern == FAN_IN:
            num_tx = len(self.accounts) - 1
            assert num_tx == len(self.scheduling_times), \
                "Mismatch between num_tx " + str(num_tx) + " and scheduling times len " + str(len(self.scheduling_times)) + " in Fan-in"

            single_amt = self._calculate_amount(self.amount / num_tx)

            random.shuffle(self.scheduling_times)
            destination = random.choice(list(self.accounts))
            sources = self.accounts - destination

            for source, t in zip(sources, self.scheduling_times):
                tx = (source, destination, single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == FAN_OUT:
            num_tx = len(self.accounts) - 1
            assert num_tx == len(self.scheduling_times), \
                "Mismatch between num_tx " + str(num_tx) + " and scheduling times len " + str(len(self.scheduling_times)) + " in Fan-out"

            single_amt = self._calculate_amount(self.amount / num_tx)

            random.shuffle(self.scheduling_times)
            source = random.choice(list(self.accounts))
            destinations = self.accounts - source
            for destination, t in zip(destinations, self.scheduling_times):
                tx = (source, destination, single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == CYCLE:
            num_tx = len(self.accounts)
            assert num_tx == len(self.scheduling_times), \
                "Mismatch between num_tx " + str(num_tx) + " and scheduling times len " + str(len(self.scheduling_times)) + " in Cycle"

            single_amt = self._calculate_amount(self.amount)

            accounts = list(self.accounts)
            random.shuffle(accounts)

            for i, t in enumerate(self.scheduling_times[:-2]):
                tx = (accounts[i], accounts[i + 1], single_amt, t)
                self.transactions[t].append(tx)

            time = self.scheduling_times[-1]
            tx = (accounts[-1], accounts[0], single_amt, time)
            self.transactions[time].append(tx)

        elif self.pattern == SCATTER_GATHER:
            num_tx = 2 * (len(self.accounts) - 2)
            assert num_tx == len(self.scheduling_times), \
                "Mismatch between num_tx " + str(num_tx) + " and scheduling times len " + str(len(self.scheduling_times)) + " in Scatter-Gather"

            single_amt = self._calculate_amount(self.amount / (num_tx/2))

            accounts = self.accounts
            source = random.choice(list(accounts))
            destination = random.choice(list(accounts - source))
            accounts = list((accounts - source) - destination)
            random.shuffle(accounts)

            first_half = self.scheduling_times[0:num_tx / 2]
            for acc, t in zip(accounts, first_half):
                tx = (source, acc, single_amt, t)
                self.transactions[t].append(tx)

            second_half = self.scheduling_times[num_tx / 2:]
            for acc, t in zip(accounts, second_half):
                tx = (acc, destination, single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == GATHER_SCATTER:
            num_tx = len(self.accounts) - 1
            assert num_tx == len(self.scheduling_times), \
                "Mismatch between num_tx " + str(num_tx) + " and scheduling times len " + str(len(self.scheduling_times)) + " in Gather-Scatter"

            num_sources, num_layerer, num_dest = self.get_account_distribution(len(self.accounts))
            remaining_accounts = self.accounts.copy()
            sources = random.sample(list(remaining_accounts), k=num_sources)
            remaining_accounts = remaining_accounts - set(sources)
            layerer = random.choice(remaining_accounts)
            remaining_accounts = remaining_accounts - set(layerer)
            destinations = remaining_accounts

            single_amt_in = self.amount / num_sources
            single_amt = self._calculate_amount(single_amt_in)

            first_half = self.scheduling_times[0:num_tx / 2]
            for source, t in zip(sources, first_half):
                tx = (source, layerer, single_amt, t)
                self.transactions[t].append(tx)

            single_amt_out = self.amount / num_dest
            single_amt = self._calculate_amount(single_amt_out)

            second_half = self.scheduling_times[num_tx / 2:]
            for destination, t in zip(destinations, second_half):
                tx = (layerer, destination, single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == U:
            num_tx = (len(self.accounts) + 1) * 2 - 2
            assert num_tx == len(self.scheduling_times), \
                "Mismatch between num_tx " + str(num_tx) + " and scheduling times len " + str(len(self.scheduling_times)) + " in U"

            single_amt = self._calculate_amount(self.amount)

            accounts = list(self.accounts)
            random.shuffle(accounts)

            first_half = self.scheduling_times[0:num_tx / 2]
            for i, t in enumerate(first_half):
                tx = (accounts[i], accounts[i + 1], single_amt, t)
                self.transactions[t].append(tx)

            second_half = self.scheduling_times[num_tx / 2:]
            for i, t in enumerate(second_half):
                tx = (accounts[-i - 1], accounts[-i - 2], single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == REPEATED:
            num_tx = len(self.scheduling_times)

            single_amt = self.amount / num_tx
            single_amt = self._calculate_amount(single_amt)

            source = list(self.accounts)[0]
            dest = list(self.accounts)[1]

            for t in self.scheduling_times:
                tx = (source, dest, single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == CASH_IN:
            num_tx = self.get_num_txs_by_type()

            single_amt = self.amount / num_tx
            single_amt = self._calculate_amount(single_amt)

            for t in self.scheduling_times:
                tx = (None, list(self.accounts)[0], single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == CASH_OUT:
            num_tx = self.get_num_txs_by_type()

            single_amt = self.amount / num_tx
            single_amt = self._calculate_amount(single_amt)

            for t in self.scheduling_times:
                tx = (list(self.accounts)[0], None, single_amt, t)
                self.transactions[t].append(tx)


class StructuredPattern(Pattern):
    def __init__(self, pattern_id, pattern_type, period, amount, is_aml, rounded_ratio, under_threshold_ratio, sources=None, layers=None, destinations=None,
                 scheduling_type=RANDOM):
        super(StructuredPattern, self).__init__(pattern_id, pattern_type, period, amount, is_aml,
                                                rounded_ratio=rounded_ratio, under_threshold_ratio=under_threshold_ratio,
                                                scheduling_type=scheduling_type)

        self.sources = sources if sources is not None else set()
        self.layerers = layers if layers is not None else set()
        self.destinations = destinations if destinations is not None else set()

    def add_source(self, acct_id):
        self.sources.add(acct_id)

    def add_layerer(self, acct_id):
        self.layerers.add(acct_id)

    def add_destination(self, acct_id):
        self.destinations.add(acct_id)

    def schedule(self, max_time):
        start_time = random.randint(0, max_time-self.period-1)
        end_time = start_time + self.period

        num_tx = self.get_num_txs_by_type(len(self.sources), len(self.layerers), len(self.destinations))

        self._schedule(num_tx, start_time, end_time)

    def schedule_tx(self):
        if self.pattern == FAN_IN:
            num_tx = len(self.sources)
            assert num_tx == len(self.scheduling_times), \
                "Mismatch between num_tx " + str(num_tx) + " and scheduling times len " + str(len(self.scheduling_times)) + " in AML-Fan-in"

            single_amt = self.amount / num_tx
            single_amt = self._calculate_amount(single_amt)

            random.shuffle(self.scheduling_times)
            for source, t in zip(self.sources, self.scheduling_times):
                tx = (source, self.destinations[0], single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == FAN_OUT:
            num_tx = len(self.destinations)
            assert num_tx == len(self.scheduling_times), \
                "Mismatch between num_tx " + str(num_tx) + " and scheduling times len " + str(len(self.scheduling_times)) + " in AML-Fan-out"

            single_amt = self.amount / num_tx
            single_amt = self._calculate_amount(single_amt)

            random.shuffle(self.scheduling_times)
            for destination, t in zip(self.destinations, self.scheduling_times):
                tx = (self.sources[0], destination, single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == CYCLE:
            num_tx = len(self.layerers)
            assert num_tx == len(self.scheduling_times), \
                "Mismatch between num_tx " + str(num_tx) + " and scheduling times len " + str(len(self.scheduling_times)) + " in AML-Cycle"

            single_amt = self._calculate_amount(self.amount)

            random.shuffle(self.layerers)

            for i, t in enumerate(self.scheduling_times[:-2]):
                tx = (self.layerers[i], self.layerers[i + 1], single_amt, t)
                self.transactions[t].append(tx)
            time = self.scheduling_times[-1]
            tx = (self.layerers[-1], self.layerers[0], single_amt, time)
            self.transactions[time].append(tx)

        elif self.pattern == SCATTER_GATHER:
            num_tx = 2 * len(self.layerers)
            assert num_tx == len(self.scheduling_times), \
                "Mismatch between num_tx " + str(num_tx) + " and scheduling times len " + str(len(self.scheduling_times)) + " in AML-Scatter-Gather"

            single_amt = self.amount / (num_tx / 2)
            single_amt = self._calculate_amount(single_amt)

            first_half = self.scheduling_times[0:num_tx/2]
            for layer, t in zip(self.layerers, first_half):
                tx = (self.sources[0], layer, single_amt, t)
                self.transactions[t].append(tx)

            second_half = self.scheduling_times[num_tx / 2:]
            for layer, t in zip(self.layerers, second_half):
                tx = (layer, self.destinations[0], single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == GATHER_SCATTER:
            num_tx = len(self.sources) + len(self.destinations)
            assert num_tx == len(self.scheduling_times), \
                "Mismatch between num_tx " + str(num_tx) + " and scheduling times len " + str(len(self.scheduling_times)) + " in AML-Gather-Scatter"

            single_amt_in = self.amount / len(self.sources)
            single_amt = self._calculate_amount(single_amt_in)

            first_half = self.scheduling_times[0:num_tx / 2]
            for source, t in zip(self.sources, first_half):
                tx = (source, self.layerers[0], single_amt, t)
                self.transactions[t].append(tx)

            single_amt_out = self.amount / len(self.destinations)
            single_amt = self._calculate_amount(single_amt_out)

            second_half = self.scheduling_times[num_tx / 2:]
            for destination, t in zip(self.destinations, second_half):
                tx = (self.layerers[0], destination, single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == U:
            num_tx = (len(self.layerers) + 1) * 2 - 2
            assert num_tx == len(self.scheduling_times), \
                "Mismatch between num_tx " + str(num_tx) + " and scheduling times len " + str(len(self.scheduling_times)) + " in AML-U"

            single_amt = self._calculate_amount(self.amount)

            first_half = self.scheduling_times[0:num_tx / 2]
            for i, t in enumerate(first_half):
                tx = (self.layerers[i], self.layerers[i + 1], single_amt, t)
                self.transactions[t].append(tx)

            second_half = self.scheduling_times[num_tx / 2:]
            for i, t in enumerate(second_half):
                tx = (self.layerers[-i - 1], self.layerers[-i - 2], single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == REPEATED:
            num_tx = len(self.scheduling_times)

            single_amt = self.amount / num_tx
            single_amt = self._calculate_amount(single_amt)

            for t in self.scheduling_times:
                tx = (self.sources[0], self.destinations[0], single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == BIPARTITE:
            min_layers_dim = int(len(self.layerers) / BIPARTITE_MAX_LAYERS_NUM)
            max_layers_dim = int(len(self.layerers) / BIPARTITE_MIN_LAYERS_NUM)

            layerer = self.layerers.copy()
            layers = random_chunk(layerer, min_chunk=min_layers_dim, max_chunk=max_layers_dim)
            layers.append(list(self.destinations).copy())
            layers.insert(0, list(self.sources).copy())

            # For each couple of layers create a bipartite graph
            edge_id = 0
            for i in range(0, len(layers)-1):
                senders = layers[i]
                receivers = layers[i+1]

                edges = []
                for sender in senders:
                    for receiver in receivers:
                        if random.random() < BIPARTITE_EDGE_DENSITY:
                            edges.append((sender, receiver))
                num_txs = len(edges)

                single_amt = self.amount / num_txs
                single_amt = self._calculate_amount(single_amt)

                for edge in edges:
                    (a, b) = edge
                    time = self.scheduling_times[edge_id]
                    tx = (a, b, single_amt, time)
                    self.transactions[time].append(tx)
                    edge_id += 1

        elif self.pattern == CASH_IN:
            num_tx = self.get_num_txs_by_type()

            single_amt = self.amount / num_tx
            single_amt = self._calculate_amount(single_amt)

            for t in self.scheduling_times:
                tx = (None, self.destinations[0], single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == CASH_OUT:
            num_tx = self.get_num_txs_by_type()

            single_amt = self.amount / num_tx
            single_amt = self._calculate_amount(single_amt)

            for t in self.scheduling_times:
                tx = (self.sources[0], None, single_amt, t)
                self.transactions[t].append(tx)

        else:
            raise NotImplementedError
