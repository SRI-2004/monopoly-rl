import numpy as np

class SyncVecEnv:
    """
    A simple, synchronous vectorized environment wrapper for PettingZoo ParallelEnv.
    """
    def __init__(self, env_fns):
        """
        Args:
            env_fns (list[callable]): A list of functions that create the environments.
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.metadata = self.envs[0].metadata
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.possible_agents = self.envs[0].possible_agents

    def reset(self, seed=None):
        """
        Resets all environments.
        """
        obs_list = []
        infos_list = []
        for i, env in enumerate(self.envs):
            # Pass a different seed to each environment
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            obs_list.append(obs)
            infos_list.append(info)
        return self._stack_obs(obs_list), infos_list

    def step(self, actions):
        """
        Steps all environments with their respective actions.

        Args:
            actions (list[dict]): A list of action dictionaries, one for each environment.
        """
        obs_list, rews_list, terms_list, truncs_list, infos_list = [], [], [], [], []
        
        for i, env in enumerate(self.envs):
            obs, rew, term, trunc, info = env.step(actions[i])
            
            # If all agents in an env are done, reset it
            if not env.agents:
                obs, info = env.reset()

            obs_list.append(obs)
            rews_list.append(rew)
            terms_list.append(term)
            truncs_list.append(trunc)
            infos_list.append(info)
        
        return (
            self._stack_obs(obs_list),
            self._stack_rewards(rews_list),
            self._stack_dones(terms_list),
            self._stack_dones(truncs_list),
            infos_list,
        )
    
    def _stack_obs(self, obs_list):
        """
        Stacks observations from multiple environments.
        Converts [{agent: obs}, {agent: obs}] -> {agent: [obs, obs]}
        """
        stacked_obs = {agent: [] for agent in self.possible_agents}
        for obs_dict in obs_list:
            for agent, obs in obs_dict.items():
                stacked_obs[agent].append(obs)
        
        # Now, stack the inner lists of observation dicts into numpy arrays
        for agent, agent_obs_list in stacked_obs.items():
            if agent_obs_list: # If there are observations for this agent
                # From list of dicts to dict of lists
                obs_keys = agent_obs_list[0].keys()
                dict_of_lists = {key: [d[key] for d in agent_obs_list] for key in obs_keys}
                # From dict of lists to dict of numpy arrays
                stacked_obs[agent] = {key: np.array(val) for key, val in dict_of_lists.items()}

        return stacked_obs

    def _stack_rewards(self, rewards_list):
        """
        Stacks rewards.
        Converts [{agent: rew}, {agent: rew}] -> {agent: [rew, rew]}
        """
        stacked_rewards = {agent: [] for agent in self.possible_agents}
        for reward_dict in rewards_list:
            for agent, reward in reward_dict.items():
                stacked_rewards[agent].append(reward)
        
        for agent, rew_list in stacked_rewards.items():
            stacked_rewards[agent] = np.array(rew_list)
            
        return stacked_rewards

    def _stack_dones(self, dones_list):
        """
        Stacks termination or truncation flags.
        """
        stacked_dones = {agent: [] for agent in self.possible_agents}
        for done_dict in dones_list:
            for agent, done in done_dict.items():
                stacked_dones[agent].append(done)

        for agent, done_list in stacked_dones.items():
            stacked_dones[agent] = np.array(done_list)

        return stacked_dones

    def close(self):
        """
        Closes all environments.
        """
        for env in self.envs:
            env.close() 