import multiprocessing as mp
import numpy as np
from rl_agent.env_wrapper import preprocess_obs # Import the preprocessing function

def worker(remote, parent_remote, env_fn, seed):
    """
    The worker process that runs a single environment instance.
    This is now responsible for preprocessing observations before sending them back.
    """
    parent_remote.close()
    env = env_fn()
    env.reset(seed=seed)
    num_agents = len(env.possible_agents)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(data)
                
                # Preprocess observations for each agent
                processed_obs = np.array([preprocess_obs(obs_dict[agent], num_agents, i) for i, agent in enumerate(env.possible_agents)])
                
                # Check if the episode is done for any agent
                is_done = any(term_dict.values()) or any(trunc_dict.values())
                
                if is_done:
                    # If done, the info_dict from the env contains the final info.
                    # We need to send this back with a special key. The env will auto-reset.
                    final_obs_dict, _ = env.reset()
                    processed_final_obs = np.array([preprocess_obs(final_obs_dict[agent], num_agents, i) for i, agent in enumerate(env.possible_agents)])
                    remote.send((processed_final_obs, reward_dict, term_dict, trunc_dict, {'final_info': info_dict}))
                else:
                    remote.send((processed_obs, reward_dict, term_dict, trunc_dict, info_dict))
            elif cmd == 'reset':
                obs_dict, info_dict = env.reset()
                processed_obs = np.array([preprocess_obs(obs_dict[agent], num_agents, i) for i, agent in enumerate(env.possible_agents)])
                remote.send((processed_obs, info_dict))
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()

class MultiProcessingVecEnv:
    """
    A truly parallel vectorized environment that offloads observation preprocessing
    to the worker processes, sending back clean numerical arrays.
    """
    def __init__(self, env_fns, seed):
        self.num_envs = len(env_fns)
        self.waiting = False
        self.closed = False
        
        ctx = mp.get_context('fork')
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        for i, (work_remote, remote, env_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns)):
            process = ctx.Process(target=worker, args=(work_remote, remote, env_fn, seed + i))
            process.daemon = True
            process.start()
            self.processes.append(process)
            work_remote.close()
        
        # Get spaces and agent info from one of the envs
        temp_env = env_fns[0]()
        self.possible_agents = temp_env.possible_agents
        self.num_agents = len(self.possible_agents)
        
        # The observation space is now the preprocessed, flattened space
        sample_obs_dict = temp_env.observation_space("player_0").sample()
        self.single_observation_space_shape = preprocess_obs(sample_obs_dict, self.num_agents, 0).shape
        self.observation_space_shape = (self.num_envs, self.num_agents, *self.single_observation_space_shape)
        
        self.action_space = temp_env.action_space
        temp_env.close()

    def step_async(self, actions):
        # Actions are expected to be a dict of arrays, e.g., {'player_0': [action_env0, action_env1, ...]}
        # We need to transpose this to a list of dicts for each environment
        actions_per_env = [{} for _ in range(self.num_envs)]
        for agent_id, agent_actions in actions.items():
            for i, action in enumerate(agent_actions):
                actions_per_env[i][agent_id] = action

        for remote, action_dict in zip(self.remotes, actions_per_env):
            remote.send(('step', action_dict))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, terms, truncs, infos = zip(*results)
        
        # Stack the pre-processed observations from all environments
        stacked_obs = np.stack(obs)
        
        # The 'infos' tuple from the workers is already in the correct format.
        # The consumer (evaluate.py) is responsible for checking for 'final_info'.
        # No reformatting is needed here.

        return (
            stacked_obs,
            self._stack_dicts(rews),
            self._stack_dicts(terms),
            self._stack_dicts(truncs),
            infos, # Pass the raw infos tuple directly through
        )

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        
        # Stack the pre-processed observations
        stacked_obs = np.stack(obs)
        return stacked_obs, infos

    def close(self):
        if self.closed:
            return
        if self.waiting:
            self.step_wait()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def _stack_dicts(self, dict_list):
        """Stacks a list of dictionaries into a single dictionary of numpy arrays."""
        stacked_dict = {agent: [] for agent in self.possible_agents}
        for d in dict_list:
            for agent, value in d.items():
                stacked_dict[agent].append(value)
        
        for agent, values in stacked_dict.items():
            stacked_dict[agent] = np.array(values)
            
        return stacked_dict 