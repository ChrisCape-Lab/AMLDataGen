import random
import networkx as nx
import logging

from src.model.population import Account
from src.utils import get_degrees

import src._constants as _c
from src._variables import COMMUNITY


class CommunityNode:
    def __init__(self, node_id, capacity, patterns_list, avg_fan_out, avg_fan_in):
        self.id = node_id
        self.capacity = capacity
        self.patterns = patterns_list
        self.avg_fan_out = avg_fan_out
        self.avg_fan_in = avg_fan_in

    def attributes_to_dict(self):
        return {'capacity': self.capacity}


class Community:
    def __init__(self):
        self.connection_graph = nx.DiGraph()
        self.communities = list()
        self.community_to_account = dict()
        self.nodes = dict()

    # GETTERS
    # ------------------------------------------

    def get_nodes_ids(self):
        nodes_ids = list()
        for _, node in self.nodes.items():
            nodes_ids.append(node.id)

        return nodes_ids

    def get_sources_for(self, node_id: int) -> list:
        if self.connection_graph.is_directed():
            return self.connection_graph.successors(node_id)
        else:
            return self.get_neighbours_for(node_id)

    def get_destinations_for(self, node_id: int) -> list:
        if self.connection_graph.is_directed():
            return self.connection_graph.predecessors(node_id)
        else:
            return self.get_neighbours_for(node_id)

    def get_random_destination_for(self, node_id: int, new_node_prop: float) -> int:
        node_destinations = self.get_destinations_for(node_id)
        new_destination = random.random() > new_node_prop
        if not new_destination:
            return random.choice(node_destinations)
        else:
            new_available_destinations = set(self.get_nodes_ids()) - set(node_destinations)
            return random.choice(list(new_available_destinations))

    def get_neighbours_for(self, node_id: int) -> list:
        return self.connection_graph.neighbors(node_id)

    def get_unknown_nodes_for(self, node_id: int) -> list:
        return list(set(self.connection_graph.nodes()) - set(self.get_destinations_for(node_id)))

    def get_fan_in_list(self):
        fan_in_list = list()
        for node_id in self.get_nodes_ids():
            fan_in_list.append(self.get_sources_for(node_id))

        return fan_in_list

    def get_fan_out_list(self):
        fan_out_list = list()
        for node_id in self.get_nodes_ids():
            fan_out_list.append(self.get_destinations_for(node_id))

        return fan_out_list

    # SETTERS
    # ------------------------------------------

    def add_node(self, node: Account):
        self.nodes[node.id] = CommunityNode(node.id, node.available_balance, node.behaviours, node.avg_tx_per_step, None)

    def add_link(self, source: int, destination: int) -> None:
        if destination in self.get_destinations_for(source):
            return

        self.connection_graph.add_edge(source, destination)

    def update_attributes(self):
        attr_dict = dict()
        for node in self.nodes:
            attr_dict[node.id] = node.capacity

        nx.set_node_attributes(self.connection_graph, name="capacity", values=attr_dict)

    # INITIALIZERS
    # ------------------------------------------
    def create_community(self, community_type: int = _c.COMMUNITY.FULL_RANDOM, directed: bool = True, deg_file: str = ''):
        if community_type == _c.COMMUNITY.FULL_RANDOM:
            self.create_full_random_connections(directed)
        elif community_type == _c.COMMUNITY.RANDOM:
            self.create_random_communities(directed)
        elif community_type == _c.COMMUNITY.STRUCTURED_RANDOM:
            self.create_random_structured_communities(directed)
        elif community_type == _c.COMMUNITY.FROM_FILE:
            self.load_communities_from_deg_file(deg_file)
        else:
            raise NotImplementedError

    def create_full_random_connections(self, directed=True):
        """Creates a fully random connection among accounts, ignoring both business and fan_out"""
        connection_number = (len(self.nodes) ** 2) * COMMUNITY.DEF_RND_COMMUNITY_DENSITY

        if directed:
            connection_number = connection_number * 2

        self.connection_graph = nx.gnm_random_graph(n=len(self.nodes), m=connection_number, directed=directed)

    def create_random_communities(self, directed=True):
        """Creates random connections among accounts: choose an account and get n random known nodes, ignoring business"""
        if not directed:
            self.connection_graph = nx.Graph()

        self.connection_graph.add_nodes_from(range(0, len(self.nodes)))

        for node in self.nodes:
            initial_known_nodes_num = max(random.sample(range(COMMUNITY.DEF_MIN_KNOWN_NODES, COMMUNITY.DEF_MAX_KNOWN_NODES),
                                                        k=node.avg_fan_out))
            known_nodes = random.sample(self.nodes, k=initial_known_nodes_num)

            for known_node in known_nodes:
                self.connection_graph.add_edge(known_node.id, known_node)

    def create_random_structured_communities(self, directed=False):
        # Creates nodes community
        remaining_nodes = set(range(0, len(self.nodes)))
        community_id = 0
        while len(remaining_nodes) != 0:
            community_dim = random.randint(COMMUNITY.DEF_MIN_COMM_SIZE, COMMUNITY.DEF_MAX_COMM_SIZE)
            try:
                community_nodes = random.sample(remaining_nodes, k=community_dim)
            except ValueError:
                break

            # Create a community among those nodes
            new_community = nx.Graph().add_nodes_from(community_nodes)
            self.communities[community_id] = new_community
            community_id += 1

            # Remove used nodes availability
            remaining_nodes = remaining_nodes - set(community_nodes)

        # Assign remaining nodes to communities
        for node in remaining_nodes:
            chosen_community = random.randint(0, len(self.communities))
            self.communities[chosen_community].add_node(node)

        # Creating intra-communities connections

        # Creating inter-community connections

    def load_communities_from_deg_file(self, deg_file):
        """
        This functions simply creates the users connection from a degree file. This is the same approach of AMLSim, inserted for compatibility
        :param deg_file: string, is the name of the degree csv file
        :return: -
        """
        in_deg, out_deg = get_degrees(deg_file, len(self.nodes))

        if not sum(in_deg) == sum(out_deg):
            raise nx.NetworkXError('Invalid degree sequences. Sequences must have equal sums.')

        n_in = len(in_deg)
        n_out = len(out_deg)
        if n_in < n_out:
            in_deg.extend((n_out - n_in) * [0])
        else:
            out_deg.extend((n_in - n_out) * [0])

        num_nodes = len(in_deg)
        # TODO: there was a nx.MultiDiGraph, why?
        _g = nx.empty_graph(num_nodes, nx.DiGraph())
        if num_nodes == 0 or max(in_deg) == 0:
            raise nx.NetworkXError('The network is not correctly formulated!')

        in_tmp_list = list()
        out_tmp_list = list()
        for n in _g.nodes():
            in_tmp_list.extend(in_deg[n] * [n])
            out_tmp_list.extend(out_deg[n] * [n])
        random.shuffle(in_tmp_list)
        random.shuffle(out_tmp_list)

        num_edges = len(in_tmp_list)
        for i in range(num_edges):
            _src = out_tmp_list[i]
            _dst = in_tmp_list[i]
            if _src == _dst:  # ID conflict causes self-loop
                for j in range(i + 1, num_edges):
                    if _src != in_tmp_list[j]:
                        in_tmp_list[i], in_tmp_list[j] = in_tmp_list[j], in_tmp_list[i]  # Swap ID
                        break
        _g.add_edges_from(zip(out_tmp_list, in_tmp_list))

        for idx, (_src, _dst) in enumerate(_g.edges()):
            if _src == _dst:
                logging.warning("Self loop from/to %d at %d" % (_src, idx))

        self.connection_graph = _g

        logging.info("Add %d base transactions" % self.connection_graph.number_of_edges())
        nodes = self.connection_graph.nodes()

        edge_id = 0
        for src_i, dst_i in self.connection_graph.edges():
            src = nodes[src_i]
            dst = nodes[dst_i]
            self.connection_graph.edge[src][dst]['edge_id'] = edge_id
            edge_id += 1
