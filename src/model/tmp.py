


class tmp:
    def __init__(self):
        self.community =
        self.accounts = dict()
        self.bank_to_acc = dict()
        self.compromising_accounts_ratio = list()
        self.compromised = set()
        self.aml_sources = []
        self.aml_layerer = []
        self.aml_destinations = []

    # CREATION
    # ------------------------------------------

    def create_communities(self, community_type, directed, communities_file='None'):
        if community_type == COMMUNITY.FULL_RANDOM:
            .create_full_random_connections(directed=directed)
        elif community_type == COMMUNITY.RANDOM:
            self.community.create_random_communities(directed=directed)
        elif community_type == COMMUNITY.STRUCTURED_RANDOM:
            self.community.create_random_structured_communities(directed=directed)
        elif community_type == COMMUNITY.FROM_FILE:
            self.community.load_communities_from_deg_file(communities_file)
        else:
            raise NotImplementedError