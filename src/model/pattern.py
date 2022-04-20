import random
import networkx as nx

from src._constants import *
from src.utils import random_chunk


def get_rounded(value):
    if value > 0:
        amount = round(value, 2)
        digits = len(str(amount))
        most_significant_nums = str(amount)[0:int(digits/2)]
        rounded_amt = str(most_significant_nums).ljust(digits, '0')

        return rounded_amt
    else:
        print("Negative Amount!")


class Pattern:
    def __init__(self, pattern_id, pattern_type, period, amount, scheduling_type=RANDOM):
        self.id = pattern_id
        self.pattern = pattern_type
        self.scheduling_times = list()
        self.scheduling_type = scheduling_type
        self.period = period
        self.amount = amount
        self.transactions = dict()

    def get_account_distribution(self, num_accounts):
        # For Scatter-Gather, determine the number of sources in order to set the number of destinations
        src = random.choice(range(0, num_accounts-1))

        # For Bipartite, determine the number of layers and the sources/destinations distribution
        num_sources = int(random.uniform(0.20, 0.30) * num_accounts)
        num_destinations = int(random.uniform(0.20, 0.30) * num_accounts)

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

    def schedule(self, max_time):
        raise NotImplementedError

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

    def update(self, time):
        self.scheduling_times = self.scheduling_type - time
        del self.transactions[time]


class NormalPattern(Pattern):
    def __init__(self, pattern_id, pattern_type, period, amount, accounts, scheduling_type=RANDOM):
        super(NormalPattern, self).__init__(pattern_id, pattern_type, period, amount, scheduling_type=scheduling_type)

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

            single_amt = round(self.amount / num_tx, 2)

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

            single_amt = round(self.amount / num_tx, 2)

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

            single_amt = round(self.amount, 2)

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

            single_amt = round(self.amount / (num_tx/2), 2)

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
                "Mismatch between num_tx " + str(num_tx) + " and scheduling times len " + str(len(self.scheduling_times)) + " in AML-Gather-Scatter"

            single_amt_in = (self.amount / 2) / len(self.sources)
            single_amt = get_rounded(single_amt_in)

            first_half = self.scheduling_times[0:num_tx / 2]
            for source, t in zip(self.sources, first_half):
                tx = (source, self.layers[0], single_amt, t)
                self.transactions[t].append(tx)

            single_amt_out = (self.amount / 2) / len(self.destinations)
            single_amt = get_rounded(single_amt_out)

            second_half = self.scheduling_times[num_tx / 2:]
            for destination, t in zip(self.destinations, second_half):
                tx = (self.layers[0], destination, single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == U:
            num_tx = (len(self.accounts) + 1) * 2 - 2
            assert num_tx == len(self.scheduling_times), \
                "Mismatch between num_tx " + str(num_tx) + " and scheduling times len " + str(len(self.scheduling_times)) + " in U"

            single_amt = round(self.amount, 2)

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
            single_amt = round(single_amt, 2)

            source = list(self.accounts)[0]
            dest = list(self.accounts)[1]

            for t in self.scheduling_times:
                tx = (source, dest, single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == CASH_IN:
            num_tx = self.get_num_txs_by_type()

            single_amt = self.amount / num_tx
            single_amt = round(single_amt, 2)

            for t in self.scheduling_times:
                tx = (None, list(self.accounts)[0], single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == CASH_OUT:
            num_tx = self.get_num_txs_by_type()

            single_amt = self.amount / num_tx
            single_amt = get_rounded(single_amt)

            for t in self.scheduling_times:
                tx = (list(self.accounts)[0], None, single_amt, t)
                self.transactions[t].append(tx)


class AMLPattern(Pattern):
    def __init__(self, pattern_id, pattern_type, period, amount, sources=None, layers=None, destinations=None, scheduling_type=RANDOM):
        super(AMLPattern, self).__init__(pattern_id, pattern_type, period, amount, scheduling_type=scheduling_type)

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
            single_amt = get_rounded(single_amt)

            random.shuffle(self.scheduling_times)
            for source, t in zip(self.sources, self.scheduling_times):
                tx = (source, self.destinations[0], single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == FAN_OUT:
            num_tx = len(self.destinations)
            assert num_tx == len(self.scheduling_times), \
                "Mismatch between num_tx " + str(num_tx) + " and scheduling times len " + str(len(self.scheduling_times)) + " in AML-Fan-out"

            single_amt = self.amount / num_tx
            single_amt = get_rounded(single_amt)

            random.shuffle(self.scheduling_times)
            for destination, t in zip(self.destinations, self.scheduling_times):
                tx = (self.sources[0], destination, single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == CYCLE:
            num_tx = len(self.layerers)
            assert num_tx == len(self.scheduling_times), \
                "Mismatch between num_tx " + str(num_tx) + " and scheduling times len " + str(len(self.scheduling_times)) + " in AML-Cycle"

            single_amt = get_rounded(self.amount)

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
            single_amt = get_rounded(single_amt)

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
            single_amt = get_rounded(single_amt_in)

            first_half = self.scheduling_times[0:num_tx / 2]
            for source, t in zip(self.sources, first_half):
                tx = (source, self.layerers[0], single_amt, t)
                self.transactions[t].append(tx)

            single_amt_out = self.amount / len(self.destinations)
            single_amt = get_rounded(single_amt_out)

            second_half = self.scheduling_times[num_tx / 2:]
            for destination, t in zip(self.destinations, second_half):
                tx = (self.layerers[0], destination, single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == U:
            num_tx = (len(self.layerers) + 1) * 2 - 2
            assert num_tx == len(self.scheduling_times), \
                "Mismatch between num_tx " + str(num_tx) + " and scheduling times len " + str(len(self.scheduling_times)) + " in AML-U"

            single_amt = get_rounded(self.amount)

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
            single_amt = get_rounded(single_amt)

            for t in self.scheduling_times:
                tx = (self.sources[0], self.destinations[0], single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == BIPARTITE:
            min_layers_dim = int(len(self.layerers) / 6)
            max_layers_dim = int(len(self.layerers) / 2)

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
                        if random.random() > 0.2:
                            edges.append((sender, receiver))
                num_txs = len(edges)

                single_amt = self.amount / num_txs
                single_amt = get_rounded(single_amt)

                for edge in edges:
                    (a, b) = edge
                    time = self.scheduling_times[edge_id]
                    tx = (a, b, single_amt, time)
                    self.transactions[time].append(tx)
                    edge_id += 1

        elif self.pattern == CASH_IN:
            num_tx = self.get_num_txs_by_type()

            single_amt = self.amount / num_tx
            single_amt = get_rounded(single_amt)

            for t in self.scheduling_times:
                tx = (None, self.destinations[0], single_amt, t)
                self.transactions[t].append(tx)

        elif self.pattern == CASH_OUT:
            num_tx = self.get_num_txs_by_type()

            single_amt = self.amount / num_tx
            single_amt = get_rounded(single_amt)

            for t in self.scheduling_times:
                tx = (self.sources[0], None, single_amt, t)
                self.transactions[t].append(tx)

        else:
            raise NotImplementedError
