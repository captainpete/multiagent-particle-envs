import numpy as np
from multiagent.core import World, Agent
from multiagent.scenario import BaseScenario

# 2D simple physics with agent collissions bordered by a force field.

# action space: TODO update
#   2D hybrid action space (up/down, left/right, aim left/right, fire)

# observation space:
#   Partial information on 2D positions of all agents (continuous 4D *
#   agents). This includes absolute position and velocity for the agent itself,
#   and relative position and velociry for other humans, and other zombies
#   (agents' positions are sorted by team then by distance to the observer).

# reward function:
#   When a zombie and human come into contact then 10 reward is deducted from
#   the human and awarded to the zombie for every simulation step they remain
#   in contact.

class Scenario(BaseScenario):
    def make_world(self):
        world = World()

        # set any world properties first
        num_humans = 2
        num_zombies = 5
        num_agents = num_humans + num_zombies

        # add agents
        agents = []
        # add humans' team
        for i in range(num_humans):
            agent = Agent()
            agent.collide = True
            agent.team = 0
            agent.name = 'human %d' % i
            agent.size = 0.05
            agent.accel = 4.0
            agent.max_speed = 1.3
            agent.health_decay = 0.99

            # weapons
            agent.armed = True
            agent.arms_reload_time = 1.0
            agent.arms_pallet_count = 10
            agent.arms_pallet_damage = 0.1
            agent.arms_pallet_range = 2
            agent.arms_pallet_spread = 5 /360.0*2*np.pi

            agents.append(agent)
        # add zombies' team
        for i in range(num_zombies):
            agent = Agent()
            agent.collide = True
            agent.team = 1
            agent.name = 'zombie %d' % i
            agent.size = 0.05
            agent.accel = 3.0
            agent.max_speed = 1.0
            agents.append(agent)
        world.agents = agents

        # make initial conditions
        self.reset_world(world)

        return world

    def reset_world(self, world):
        # random properties for agents
        team_colors = [
            np.array([0.35, 0.85, 0.35]),
            np.array([0.85, 0.35, 0.35])
        ]
        for i, agent in enumerate(world.agents):
            agent.color = np.array(team_colors[agent.team])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-.9, .9, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.health = 1.0
            if agent.armed:
                agent.state.aim_heading = np.random.uniform(0, 2 * np.pi)
                agent.state.aim_vel = 0

    def reward(self, agent, world):
        if agent.team == 0:
            return -10 * agent.state.biting
        elif agent.team == 1:
            return 10 * agent.state.biting
        else:
            raise "Undefined reward for team %d" % agent.team

    def observation(self, agent, world):
        agents = np.array(world.agents)

        # sort agents by distance from agent
        pos = np.array([a.state.p_pos for a in agents])
        delta_pos = pos - agent.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos), axis=1))
        agents = agents[np.argsort(dist)]

        # sort again by team
        humans = [a for a in agents if a.team == 0]
        zombies = [a for a in agents if a.team == 1]
        agents = humans + zombies

        armed_agents = [a for a in agents if a.armed]
        other_agents = [a for a in agents if a is not agent]

        # the observation space
        #
        # self:
        #
        #   pos         2
        #   vel         2
        #   ang         1
        #   health      1
        #
        # others: (sorted by team then distance, must be constant size within team)
        #
        #   rel_pos     2
        #   rel_vel     2
        #   rel_ang     1
        #   reloading   1

        s = 2+2+1 + (2+2+1+1)*len(other_agents)
        obs = np.zeros((s,))
        i = 0

        # self observation
        obs[i:i+2] = agent.state.p_pos;     i += 2
        obs[i:i+2] = agent.state.p_vel;     i += 2
        obs[i:i+1] = agent.state.health;    i += 1

        abs_ang = agent.state.aim_heading if agent.armed else 0.
        for other in other_agents:
            # relative position and velocity
            rel_pos = other.state.p_pos - agent.state.p_pos
            rel_vel = other.state.p_vel - agent.state.p_vel
            obs[i:i+2] = rel_pos; i += 2
            obs[i:i+2] = rel_vel; i += 2
            # relative angle to current heading (not relative heading)
            rel_ang = np.arctan2(rel_pos[1], rel_pos[0]) - abs_ang
            obs[i:i+1] = (rel_ang % 2*np.pi) - np.pi; i += 1
            # reloading
            rel = other.state.reloading if other.armed else 0.
            obs[i:i+1] = rel; i += 1

        assert i == s

        return obs

