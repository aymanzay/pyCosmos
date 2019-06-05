from gym.envs.registration import register

register(id='agent-v0', entry_point='agent_gym.envs: AgentEnv')