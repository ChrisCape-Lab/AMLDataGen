import random
import pandas as pd
import networkx as nx
import logging

from src.model.population import Account
from src.utils import get_degrees

import src._constants as _c
import src._variables as _v


class CommunityNode:
    """
    CommunityNode is a class that represent a node with some attributes in the community.
    In this context can be considered as an abstraction of an account which has only the attributes needed by the
    community graph, in order to perform search or matching

    # Properties
        id: int, just a unique identifier
        sched_balance: float, is the balance of the account without the scheduled-but-not-done transactions amounts
        patterns: list(int), is the list of patterns that the account can perform
        type: int, the type of the node
        avg_fan_out: float, is the average amount of transactions the ndoe can perform per time instant
        avg_fan_in: float, not used

    # Methods
        attributes_to_dict: return some properties of the node as dictionary. Useful for nx.graph creation

    """
    def __init__(self, node_id, sched_balance, patterns_list, node_type, avg_fan_out, avg_fan_in):
        self.id = node_id
        self.sched_balance = sched_balance
        self.patterns = patterns_list
        self.type = node_type
        self.avg_fan_out = avg_fan_out
        self.avg_fan_in = avg_fan_in

    def attributes_to_dict(self) -> dict:
        """
        Return the graph-related properties (sched_balance, patterns and type) as a dictionary
        :return: dict(), the graph-related properties (sched_balance, patterns and type) of the CommunityNode
        """
        return {_c.ACCTS_DF_ATTR.S_SCHED_BALANCE: self.sched_balance, 'patterns': self.patterns, 'type': self.type}


class Community:
    """
    Community is a class which can represent the connections among users. This is basically a nx.graph holder in which
    a node is an account and a link is a relation between two accounts. The concept of relation is "the accounts can
    """
    def __init__(self):
        self.connection_graph = nx.DiGraph()
        self.communities = list()
        self.community_to_account = dict()
        self.nodes = dict()

    # GETTERS
    # ------------------------------------------

    def get_nodes_ids(self):
        """

        :return:
        """
        nodes_ids = list()
        for _, node in self.nodes.items():
            nodes_ids.append(node.id)

        return nodes_ids

    def get_sources_for(self, node_id: int) -> list:
        if self.connection_graph.is_directed():
            return [x for x in self.connection_graph.successors(node_id)]
        else:
            return self.get_neighbours_for(node_id)

    def get_destinations_for(self, node_id: int) -> list:
        if self.connection_graph.is_directed():
            return [x for x in self.connection_graph.predecessors(node_id)]
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
        return [x for x in self.connection_graph.neighbors(node_id)]

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

    def get_degrees_count(self):
        comm_df = pd.DataFrame(columns=['In-degree', 'Out-degree'])
        for _, node in self.nodes.items():
            in_deg = len(self.get_sources_for(node.id))
            out_deg = len(self.get_destinations_for(node.id))
            comm_df.loc[len(comm_df.index)] = [in_deg, out_deg]
        comm_df = comm_df.groupby(['In-degree', 'Out-degree']).size().to_frame('Count').reset_index()
        comm_df = comm_df[['Count', 'In-degree', 'Out-degree']]
        comm_df = comm_df.sort_values(by=['Count', 'In-degree', 'Out-degree'], ascending=[False, False, False])

        return comm_df.values.tolist()

    # SETTERS
    # ------------------------------------------

    def add_node(self, node: Account):
        self.nodes[node.id] = CommunityNode(node.id, node.available_balance, node.behaviours, node.role, node.avg_tx_per_step, None)

    def add_link(self, source: int, destination: int) -> None:
        if destination in self.get_destinations_for(source):
            return

        self.connection_graph.add_edge(source, destination)

    def update_attributes(self):
        attributes_dict = dict()
        for n_id, node in self.nodes.items():
            attributes_dict[n_id] = node.attributes_to_dict()
        nx.set_node_attributes(self.connection_graph, attributes_dict)

    # INITIALIZERS
    # ------------------------------------------
    def create_community(self, community_type: int = _c.COMM_TYPE.FULL_RANDOM, directed: bool = True, deg_file: str = ''):
        # Create connections
        if community_type == _c.COMM_TYPE.FULL_RANDOM:
            self.create_full_random_connections(directed)
        elif community_type == _c.COMM_TYPE.RANDOM:
            self.create_random_communities(directed)
        elif community_type == _c.COMM_TYPE.STRUCTURED_RANDOM:
            self.create_random_structured_communities()
        elif community_type == _c.COMM_TYPE.FROM_FILE:
            self.load_communities_from_deg_file(deg_file)
        else:
            raise NotImplementedError

        # Add nodes attributes
        assert len(self.connection_graph.nodes) == len(self.nodes.items())
        attributes_dict = dict()
        for n_id, node in self.nodes.items():
            attributes_dict[n_id] = node.attributes_to_dict()
        nx.set_node_attributes(self.connection_graph, attributes_dict)

    def create_full_random_connections(self, directed=True):
        """Creates a fully random connection among accounts, ignoring both business and fan_out"""
        connection_number = (len(self.nodes) ** 2) * _v.COMM.DEF_RND_COMMUNITY_DENSITY

        if directed:
            connection_number = connection_number * 2

        self.connection_graph = nx.gnm_random_graph(n=len(self.nodes), m=connection_number, directed=directed)

    def create_random_communities(self, directed=True):
        """Creates random connections among accounts: choose an account and get n random known nodes, ignoring business"""
        if not directed:
            self.connection_graph = nx.Graph()

        self.connection_graph.add_nodes_from(self.get_nodes_ids())

        for _, node in self.nodes.items():
            mu = int((_v.COMM.DEF_MIN_KNOWN_NODES + _v.COMM.DEF_MAX_KNOWN_NODES)/2)
            sigma = (_v.COMM.DEF_MAX_KNOWN_NODES - mu) / 3
            initial_known_nodes_num = min(max(int(random.gauss(mu, sigma)), _v.COMM.DEF_MIN_KNOWN_NODES), _v.COMM.DEF_MAX_KNOWN_NODES)
            #initial_known_nodes_num = max(random.sample(range(_v.COMM.DEF_MIN_KNOWN_NODES, _v.COMM.DEF_MAX_KNOWN_NODES),
            #                                            k=int(max(node.avg_fan_out, 1))))
            known_nodes = random.sample(self.nodes.keys(), k=initial_known_nodes_num)

            for known_node in known_nodes:
                self.connection_graph.add_edge(node.id, known_node)

    def create_random_structured_communities(self):
        connection_number = _v.COMM.DEF_MAX_KNOWN_NODES
        self.connection_graph = nx.powerlaw_cluster_graph(n=len(self.nodes), m=connection_number, p=0.5)
        """
        # Creates nodes community
        remaining_nodes = set(range(0, len(self.nodes)))
        community_id = 0
        while len(remaining_nodes) != 0:
            community_dim = random.randint(_v.COMM.DEF_MIN_COMM_SIZE, _v.COMM.DEF_MAX_COMM_SIZE)
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
        """

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
