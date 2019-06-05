import gym
from gym import error, spaces, utils
from gym.utils import seeding
from dag import *
from graph_tool.all import *

class AgentEnv(gym.Env):

    metadata = {'render_modes': ['graph']}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    # TODO: change input parameters to account for user-inputted network
    def __init__(self, network_size=10, input_nodes=1):
        self.network_size = network_size
        self.input_nodes = input_nodes

        # TODO: Load graph from file
        self.graph = Graph()
        self.graph.set_fast_edge_removal(True)
        self.graph.add_vertex(self.network_size)

        # TODO: Change action space to current neighbor nodes + observation space to full network
        self.action_space = spaces.Tuple((spaces.Discrete(self.network_size), spaces.Discrete(self.network_size)))
        self.observation_space = spaces.MultiDiscrete(np.full((self.network_size, self.network_size), 2))
        self.time_step = 0

        #TODO: Make observation/adjacency matrix to account for past events, etc.
        self.observation = adjacency(self.graph).toarray()

        self.seed_value = self.seed()
        self.true_graph = self.create_true_graph()
        self.reset()

    def create_true_graph(self):
        # Read and create graph from existing file
        print('returning final_workflow')


    # TODO: Implement proper step function to account for agent choosing 'correct' next song.
    def step(self, action):
        done = 0
        reward = 0
        assert self.action_space.contains(action)
        valid_source_nodes = [index for index, in_degree in
                              enumerate(self.graph.get_in_degrees(self.graph.get_vertices())) if
                              ((in_degree > 0 or index < self.input_nodes) and index < (self.network_size - 1))]

        if action[0] not in valid_source_nodes:
            raise ValueError('this action does not have a valid from node')
        self.time_step += 1
        return self.observation, reward, done, {"time_step": self.time_step}

    def reset(self):
        #self.graph.clear_edges()
        self.time_step = 0
        self.observation = adjacency(self.graph).toarray()
        return self.observation

    def render(self, mode='graph'):
        if mode == 'graph':
            filename = "./renders/TrueGraphSeed" + str(self.seed_value) + ".png"
            graph_draw(self.true_graph, vertex_text=self.true_graph.vertex_index, vertex_font_size=18,
                       output_size=(1000, 1000), output=filename)
