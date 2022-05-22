import random

import networkx as nx

import src._constants as _c
import src._variables as _v
from src.utils import random_chunk, add_to_dict_of_list, addn_to_dict_of_list

NORMAL_AMT = 0
ROUNDED = 1
UNDER_THRESH = 2


def get_rounded(value):
    if value > 0:
        amount = round(value, 2)
        digits = len(str(amount))
        most_significant_nums = str(amount)[0:int(digits / 2)]
        rounded_amt = str(most_significant_nums).ljust(digits, '0')

        assert int(rounded_amt) > 0
        return int(rounded_amt)
    else:
        print("Negative Amount!")


def get_under_threshold(value):
    assert get_rounded(value) - 1 > 0
    return get_rounded(value) - 1


class Pattern:
    def __init__(self, pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio=.0,
                 under_threshold_ratio=.0, scheduling_type=_c.SCHEDULING.RANDOM):
        self.id = pattern_id
        self.pattern_type = pattern_type
        self.num_accounts = num_accounts
        self.structure = nx.DiGraph()

        self.accounts_map = dict()

        self.scheduling_type = scheduling_type
        self.period = period
        self.amount = amount
        self.rounded_ratio = rounded_ratio
        self.under_threshold_ratio = under_threshold_ratio
        self.transactions = dict()
        self.is_aml = is_aml

    # SETTERS
    # ------------------------------------------

    def add_nodes(self, nodes_map: dict) -> None:
        self.accounts_map = {**self.accounts_map, **nodes_map}

    # PRIVATE
    # ------------------------------------------

    def _get_amount(self, value):
        weights = [1 - (self.rounded_ratio + self.under_threshold_ratio), self.rounded_ratio,
                   self.under_threshold_ratio]
        case = random.choices([NORMAL_AMT, ROUNDED, UNDER_THRESH], weights=weights, k=1)[0]
        if case == NORMAL_AMT:
            return round(value, 2)
        elif case == ROUNDED:
            return get_rounded(value)
        elif case == UNDER_THRESH:
            return get_under_threshold(value)
        else:
            raise NotImplementedError

    def _schedule(self, num_tx, start_time):
        end_time = start_time + self.period

        # INSTANT SCHEDULING: each tx is executed in the same day as others
        if self.scheduling_type == _c.SCHEDULING.INSTANT:
            time = random.randint(start_time, end_time)
            scheduling_times = [time] * num_tx
        # PERIODIC SCHEDULING: each tx on a different day if possible, otherwise random sample some repeated days
        elif self.scheduling_type == _c.SCHEDULING.PERIODIC:
            if num_tx > self.period:
                scheduling_times = list(range(start_time, end_time))
                exceeding = num_tx - (end_time - start_time)
                scheduling_times = sorted(scheduling_times + sorted(random.sample(range(start_time, end_time),
                                                                                  k=exceeding)))
            else:
                scheduling_times = sorted(list(range(start_time, start_time + num_tx)))
        # RANDOM SCHEDULING: just random sample some days in the interval
        else:
            scheduling_times = sorted(random.choices(range(start_time, end_time), k=num_tx))

        return scheduling_times

    # REQUIREMENTS
    # ------------------------------------------

    def get_sources_requirements(self):
        num_sources = 0
        requirements = dict()
        for node_id, n_attr in self.structure.nodes(data=True):
            if n_attr['role'] != 's':
                continue

            node_req = dict()
            fan_out = len(self.structure.successors(node_id))
            amount = 0
            for _, e_attr in self.structure[node_id].items():
                    amount += e_attr['weight']

            node_req[_c.GENERAL.FAN_OUT_NAME] = fan_out
            node_req[_c.GENERAL.BALANCE] = amount

            requirements[node_id] = node_req
            num_sources += 1

        assert len(requirements) == num_sources

        return num_sources, requirements

    def get_layerer_requirements(self):
        num_layerers = 0
        requirements = dict()
        for node_id, n_attr in self.structure.nodes(data=True):
            if n_attr['role'] != 'l':
                continue

            node_req = dict()
            fan_out = len(self.structure.successors(node_id))
            fan_in = len(self.structure.predecessors(node_id))

            amount = 0
            for _, e_attr in self.structure[node_id].items():
                amount += e_attr['weight']
            for pre_id in self.structure.predecessors(node_id):
                amount -= self.structure[pre_id][node_id]['weight']

            node_req[_c.GENERAL.FAN_OUT_NAME] = fan_out
            node_req[_c.GENERAL.FAN_IN_NAME] = fan_in
            node_req[_c.GENERAL.BALANCE] = amount

            requirements[node_id] = node_req
            num_layerers += 1

        assert len(requirements) == num_layerers

        return num_layerers, requirements

    def get_destinations_requirements(self):
        num_destinations = 0
        requirements = dict()
        for node_id, n_attr in self.structure.nodes(data=True):
            if n_attr['role'] != 'd':
                continue

            node_req = dict()
            fan_in = len(self.structure.predecessors(node_id))
            node_req[_c.GENERAL.FAN_IN_NAME] = fan_in

            requirements[node_id] = node_req
            num_destinations += 1

        assert len(requirements) == num_destinations

        return num_destinations, requirements

    # BEHAVIOUR
    # ------------------------------------------

    def create_structure(self: int) -> None:
        raise NotImplementedError

    def schedule(self, start_time: int) -> None:
        raise NotImplementedError

    def schedule_txs(self, time: int) -> list:
        return self.transactions.get(time, None)


class RandomPattern(Pattern):
    def __init__(self, pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio=.0,
                 under_threshold_ratio=.0, scheduling_type=_c.SCHEDULING.RANDOM):
        super().__init__(pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio,
                         under_threshold_ratio, scheduling_type)

    # BEHAVIOUR
    # ------------------------------------------

    def create_structure(self):
        self.structure.add_node(0, role='s')
        self.structure.add_node(1, role='d')
        self.structure.add_edge(0, 1, weight=self._get_amount(self.amount))

    def schedule(self, start_time):
        source = self.accounts_map[0]
        destination = self.accounts_map[1]
        scheduling_time = self._schedule(1, start_time)[0]
        weight = self.structure[0][1]['weight']
        transaction = (source, destination, weight, scheduling_time, _c.GENERAL.RANDOM)
        add_to_dict_of_list(self.transactions, scheduling_time, transaction)


class FanInPattern(Pattern):
    def __init__(self, pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio=.0,
                 under_threshold_ratio=.0, scheduling_type=_c.SCHEDULING.RANDOM):
        super().__init__(pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio,
                         under_threshold_ratio, scheduling_type)

    # BEHAVIOUR
    # ------------------------------------------

    def create_structure(self):
        destination_node_id = self.num_accounts - 1
        self.structure.add_node(destination_node_id, role='d')

        num_sources = self.num_accounts - 1
        single_amount = self._get_amount(self.amount / num_sources)
        for i in range(0, num_sources):
            self.structure.add_node(i, role='s')
            self.structure.add_edge(i, destination_node_id, weight=single_amount)

    def schedule(self, start_time):
        num_tx = len(self.structure.edges())
        scheduling_times = self._schedule(num_tx, start_time)
        for i, (src, dst, attr) in enumerate(self.structure.edges(data=True)):
            source = self.accounts_map[src]
            destination = self.accounts_map[dst]
            transaction = (source, destination, attr['weight'], scheduling_times[i], _c.GENERAL.FAN_IN)
            add_to_dict_of_list(self.transactions, scheduling_times[i], transaction)


class FanOutPattern(Pattern):
    def __init__(self, pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio=.0,
                 under_threshold_ratio=.0,
                 scheduling_type=_c.SCHEDULING.RANDOM):
        super().__init__(pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio,
                         under_threshold_ratio, scheduling_type)

    # BEHAVIOUR
    # ------------------------------------------

    def create_structure(self):
        source_node_id = 0
        self.structure.add_node(source_node_id, role='s')

        num_sources = self.num_accounts - 1
        single_amount = self._get_amount(self.amount / num_sources)
        for i in range(1, num_sources + 1):
            self.structure.add_node(i, role='d')
            self.structure.add_edge(source_node_id, i, weight=single_amount)

    def schedule(self, start_time):
        num_tx = len(self.structure.edges())
        scheduling_times = self._schedule(num_tx, start_time)
        for i, (src, dst, attr) in enumerate(self.structure.edges(data=True)):
            source = self.accounts_map[src]
            destination = self.accounts_map[dst]
            transaction = (source, destination, attr['weight'], scheduling_times[i], _c.GENERAL.FAN_OUT)
            add_to_dict_of_list(self.transactions, scheduling_times[i], transaction)


class CyclePattern(Pattern):
    def __init__(self, pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio=.0,
                 under_threshold_ratio=.0, scheduling_type=_c.SCHEDULING.RANDOM):
        super().__init__(pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio,
                         under_threshold_ratio, scheduling_type)

    # BEHAVIOUR
    # ------------------------------------------

    def create_structure(self):
        single_amount = self._get_amount(self.amount)
        self.structure.add_node(0, role='l')
        for i in range(1, self.num_accounts):
            self.structure.add_node(i, role='l')
            self.structure.add_edge(i - 1, i, weight=single_amount)
        self.structure.add_edge(self.num_accounts - 1, 0, weight=single_amount)

    def schedule(self, start_time):
        num_tx = len(self.structure.edges(data=True))
        scheduling_times = self._schedule(num_tx, start_time)
        for i, (src, dst, attr) in enumerate(self.structure.edges(data=True)):
            source = self.accounts_map[src]
            destination = self.accounts_map[dst]
            transaction = (source, destination, attr['weight'], scheduling_times[i], _c.GENERAL.CYCLE)
            add_to_dict_of_list(self.transactions, scheduling_times[i], transaction)


class ScatterGatherPattern(Pattern):
    def __init__(self, pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio=.0,
                 under_threshold_ratio=.0, scheduling_type=_c.SCHEDULING.RANDOM):
        super().__init__(pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio,
                         under_threshold_ratio, scheduling_type)

    # BEHAVIOUR
    # ------------------------------------------

    def create_structure(self):
        source_node_id = 0
        self.structure.add_node(source_node_id, role='s')

        destination_node_id = self.num_accounts - 1
        self.structure.add_node(destination_node_id, role='d')

        intermediate_tx_num = self.num_accounts - 2
        single_amount = self._get_amount(self.amount / intermediate_tx_num)
        for i in range(1, intermediate_tx_num + 1):
            self.structure.add_node(i, role='l')
            self.structure.add_edge(source_node_id, i, weight=single_amount)
            self.structure.add_edge(i, destination_node_id, weight=single_amount)

    def schedule(self, start_time):
        num_tx = len(self.structure.edges())
        scheduling_times = self._schedule(num_tx, start_time)
        first_step_tx = list(filter(lambda x: x[0] == 0, self.structure.edges(data=True)))
        first_step_tx.sort(key=lambda x: x[1])
        second_step_tx = list(filter(lambda x: x[1] == self.num_accounts - 1, self.structure.edges(data=True)))
        second_step_tx.sort(key=lambda x: x[0])
        for fs_tx, ss_tx in zip(first_step_tx, second_step_tx):
            time_steps = sorted(random.sample(scheduling_times, k=2))

            source = self.accounts_map[fs_tx[0]]
            destination = self.accounts_map[fs_tx[1]]
            transaction = (source, destination, fs_tx[2]['weight'], time_steps[0], _c.GENERAL.SCATTER_GATHER)
            add_to_dict_of_list(self.transactions, time_steps[0], transaction)
            scheduling_times.remove(time_steps[0])

            source = self.accounts_map[ss_tx[0]]
            destination = self.accounts_map[ss_tx[1]]
            transaction = (source, destination, ss_tx[2]['weight'], time_steps[1], _c.GENERAL.SCATTER_GATHER)
            add_to_dict_of_list(self.transactions, time_steps[1], transaction)
            scheduling_times.remove(time_steps[1])


class GatherScatterPattern(Pattern):
    def __init__(self, pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio=.0,
                 under_threshold_ratio=.0, scheduling_type=_c.SCHEDULING.RANDOM):
        super().__init__(pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio,
                         under_threshold_ratio, scheduling_type)

    # BEHAVIOUR
    # ------------------------------------------

    def create_structure(self):
        intermediate_node_id = self.num_accounts - 1
        self.structure.add_node(intermediate_node_id, role='l')

        num_sources = random.choice(range(2, self.num_accounts - 2))
        single_amount = self._get_amount(self.amount / num_sources)
        for i in range(0, num_sources):
            self.structure.add_node(i, role='s')
            self.structure.add_edge(i, intermediate_node_id, weight=single_amount)

        num_dest = self.num_accounts - num_sources - 1
        single_amount = self._get_amount(self.amount / num_dest)
        for i in range(num_sources, num_dest + num_sources):
            self.structure.add_node(i, role='d')
            self.structure.add_edge(intermediate_node_id, i, weight=single_amount)

    def schedule(self, start_time):
        num_tx = len(self.structure.edges(data=True))
        scheduling_times = self._schedule(num_tx, start_time)
        for i, (src, dst, attr) in enumerate(self.structure.edges(data=True)):
            source = self.accounts_map[src]
            destination = self.accounts_map[dst]
            transaction = (source, destination, attr['weight'], scheduling_times[i], _c.GENERAL.GATHER_SCATTER)
            add_to_dict_of_list(self.transactions, scheduling_times[i], transaction)


class UPattern(Pattern):
    def __init__(self, pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio=.0,
                 under_threshold_ratio=.0, scheduling_type=_c.SCHEDULING.RANDOM):
        super().__init__(pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio,
                         under_threshold_ratio, scheduling_type)

    # BEHAVIOUR
    # ------------------------------------------

    def create_structure(self):
        single_amount = self._get_amount(self.amount)
        self.structure.add_node(0, role='l')
        for i in range(1, self.num_accounts):
            self.structure.add_node(i, role='l')
            self.structure.add_edge(i - 1, i, weight=single_amount)
            self.structure.add_edge(i, i - 1, weight=single_amount)

    def schedule(self, start_time):
        num_tx = len(self.structure.edges(data=True))
        scheduling_times = self._schedule(num_tx, start_time)

        tx_index = 0
        for i in range(1, int(self.num_accounts)):
            source = self.accounts_map[i - 1]
            destination = self.accounts_map[i]
            amount = self.structure[i - 1][i]['weight']
            transaction = (source, destination, amount, scheduling_times[tx_index], _c.GENERAL.U)
            add_to_dict_of_list(self.transactions, scheduling_times[i], transaction)
            tx_index += 1

        for i in range(int(self.num_accounts)-1, 0, -1):
            source = self.accounts_map[i]
            destination = self.accounts_map[i - 1]
            amount = self.structure[i - 1][i]['weight']
            transaction = (source, destination, amount, scheduling_times[tx_index], _c.GENERAL.U)
            add_to_dict_of_list(self.transactions, scheduling_times[i], transaction)
            tx_index += 1


class RepeatedPattern(Pattern):
    def __init__(self, pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio=.0,
                 under_threshold_ratio=.0, scheduling_type=_c.SCHEDULING.RANDOM):
        super().__init__(pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio,
                         under_threshold_ratio, scheduling_type)

    # BEHAVIOUR
    # ------------------------------------------

    def create_structure(self):
        self.num_accounts = 2
        num_tx = random.randint(_v.PATTERN.REPEATED_MIN, _v.PATTERN.REPEATED_MAX)
        single_amount = self._get_amount(self.amount)

        self.structure.add_node(0, role='s')
        self.structure.add_node(1, role='d')
        for i in range(0, num_tx):
            self.structure.add_edge(0, 1, weight=single_amount)

    def schedule(self, start_time):
        num_tx = len(self.structure.edges(data=True))
        scheduling_times = self._schedule(num_tx, start_time)

        for i, (src, dst, attr) in enumerate(self.structure.edges(data=True)):
            source = self.accounts_map[src]
            destination = self.accounts_map[dst]
            transaction = (source, destination, attr['weight'], scheduling_times[i], _c.GENERAL.REPEATED)
            add_to_dict_of_list(self.transactions, scheduling_times[i], transaction)


class BipartitePattern(Pattern):
    def __init__(self, pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio=.0,
                 under_threshold_ratio=.0, scheduling_type=_c.SCHEDULING.RANDOM):
        super().__init__(pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio,
                         under_threshold_ratio, scheduling_type)

    # BEHAVIOUR
    # ------------------------------------------

    def create_structure(self):
        # For Bipartite, determine the number of layers and the sources/destinations distribution
        num_sources = int(random.uniform(_v.PATTERN.BIPARTITE_MIN_SOURCES, _v.PATTERN.BIPARTITE_MAX_SOURCES) * self.num_accounts)
        num_destinations = int(
            random.uniform(_v.PATTERN.BIPARTITE_MIN_DESTINATIONS, _v.PATTERN.BIPARTITE_MAX_DESTINATIONS) * self.num_accounts)
        num_layerer = self.num_accounts - (num_sources + num_destinations)

        min_layers_dim = int(num_layerer / _v.PATTERN.BIPARTITE_MAX_LAYERS_NUM)
        max_layers_dim = int(num_layerer / _v.PATTERN.BIPARTITE_MIN_LAYERS_NUM)

        layers = list()

        # Add sources
        sources = [i for i in range(0, num_sources)]
        for i in sources:
            self.structure.add_node(i, role='s')
        layers.append(sources)

        # Add layerer
        layerer = [i for i in range(num_sources, num_layerer + num_sources)]
        for i in layerer:
            self.structure.add_node(i, role='l')
        layers.extend(list(random_chunk(layerer, min_chunk=min_layers_dim, max_chunk=max_layers_dim)))

        # Add destinations
        destinations = [i for i in range(num_layerer + num_sources, self.num_accounts)]
        for i in destinations:
            self.structure.add_node(i, role='d')
        layers.append(destinations)

        # For each couple of layers create a bipartite graph
        step = 0
        for i in range(0, len(layers) - 1):
            senders = layers[i]
            receivers = layers[i + 1]

            edges = []
            for sender in senders:
                for receiver in receivers:
                    if random.random() < _v.PATTERN.BIPARTITE_EDGE_DENSITY or len(edges) == 0:
                        edges.append((sender, receiver))
            num_txs = len(edges)

            single_amt = self.amount / num_txs
            single_amt = self._get_amount(single_amt)

            for src, dst in edges:
                self.structure.add_edge(src, dst, weight=single_amt, step=step)

            step += 1

    def schedule(self, start_time):
        num_tx = len(self.structure.edges(data=True))
        scheduling_times = self._schedule(num_tx, start_time)

        ordered_tx_list = list(self.structure.edges(data=True)).copy()
        ordered_tx_list.sort(key=lambda x: x[2]['step'])

        for i, (src, dst, attr) in enumerate(ordered_tx_list):
            source = self.accounts_map[src]
            destination = self.accounts_map[dst]
            transaction = (source, destination, attr['weight'], scheduling_times[i], _c.GENERAL.BIPARTITE)
            add_to_dict_of_list(self.transactions, scheduling_times[i], transaction)


def create_pattern(pattern_id: int, pattern_type: int, num_accounts: int, period: int, amount: float, is_aml: bool,
                   rounded_ratio: float = .0, under_threshold_ratio: float = .0,
                   scheduling_type: int = _c.SCHEDULING.RANDOM) -> Pattern:
    if pattern_type == _c.GENERAL.RANDOM:
        return RandomPattern(pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio,
                            under_threshold_ratio, scheduling_type)
    elif pattern_type == _c.GENERAL.FAN_IN:
        return FanInPattern(pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio,
                            under_threshold_ratio, scheduling_type)
    elif pattern_type == _c.GENERAL.FAN_OUT:
        return FanOutPattern(pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio,
                             under_threshold_ratio, scheduling_type)
    elif pattern_type == _c.GENERAL.CYCLE:
        return CyclePattern(pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio,
                            under_threshold_ratio, scheduling_type)
    elif pattern_type == _c.GENERAL.SCATTER_GATHER:
        return ScatterGatherPattern(pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio,
                                    under_threshold_ratio, scheduling_type)
    elif pattern_type == _c.GENERAL.GATHER_SCATTER:
        return GatherScatterPattern(pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio,
                                    under_threshold_ratio, scheduling_type)
    elif pattern_type == _c.GENERAL.U:
        return UPattern(pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio,
                        under_threshold_ratio, scheduling_type)
    elif pattern_type == _c.GENERAL.REPEATED:
        return RepeatedPattern(pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio,
                               under_threshold_ratio, scheduling_type)
    elif pattern_type == _c.GENERAL.BIPARTITE:
        return BipartitePattern(pattern_id, pattern_type, num_accounts, period, amount, is_aml, rounded_ratio,
                                under_threshold_ratio, scheduling_type)
    else:
        raise NotImplementedError
