import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
from multiagent.core import Agent

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None):

        self.world = world
        self.agents = self.world.policy_agents
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if agent.movable:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
                total_action_space.append(u_action_space)
            # arms action space
            if agent.armed:
                a_action_space = spaces.Discrete(5) # inop, left, right, trigger, safety
                total_action_space.append(a_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.a = np.zeros(2)

        # rendering
        self.viewers = [None]
        self._reset_render()

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p) # force along dimensions
        agent.action.a = np.zeros(2) # rotation, trigger-pressure
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                agent.action.u[0] += action[0][1] - action[0][2]
                agent.action.u[1] += action[0][3] - action[0][4]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]

        if agent.armed:
            # arms action
            if self.discrete_action_input:
                raise NotImplemented
            else:
                agent.action.a[0] += action[0][1] - action[0][2]
                agent.action.a[1] += action[0][3] - action[0][4]
            action = action[1:]

        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for agent in self.world.agents:
                geom = rendering.make_circle(agent.size)
                if agent.armed:
                    # render arms
                    radius0 = agent.size * 1.3
                    radius1 = agent.size * 1.5
                    aim = rendering.make_polygon([
                        (np.cos(-.1)*radius1, np.sin(-.1)*radius0),
                        (radius0, 0),
                        (np.cos(.1)*radius1, np.sin(.1)*radius0)
                    ])
                    geom = rendering.Compound([geom, aim])
                xform = rendering.Transform()
                geom.set_color(*agent.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            pos = np.zeros(self.world.dim_p)
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for a, agent in enumerate(self.world.agents):
                xform = self.render_geoms_xform[a]
                xform.set_translation(*agent.state.p_pos)
                if agent.state.aim_heading is not None:
                    xform.set_rotation(agent.state.aim_heading)

            for projectile in self.world.projectiles:
                geom = rendering.Line(start=tuple(projectile[:2]), end=tuple(projectile[2:]))
                for viewer in self.viewers:
                    viewer.add_onetime(geom)

            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx

