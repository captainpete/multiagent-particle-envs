import numpy as np
from multiagent.core import World, Agent
from multiagent.scenario import BaseScenario

# 2D simple physics with agent collissions bordered by a force field.

# action space:
#   2D continuous action space (up/down, left/right)

# observation space:
#   Complete information on 2D positions of all agents (continuous 4D *
#   agents). This includes absolute position and velocity for the agent itself,
#   and relative position and velociry for other humans, and other zombies
#   (agents have identity by index position).

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
            agent.silent = False
            agent.team = 0
            agent.name = 'human %d' % i
            agent.size = 0.05
            agent.accel = 4.0
            agent.max_speed = 1.3
            agents.append(agent)
        # add zombies' team
        for i in range(num_zombies):
            agent = Agent()
            agent.collide = True
            agent.silent = False
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
            if agent.team == 0:
                agent.state.p_pos = np.random.uniform(-.9, .9, world.dim_p)
            elif agent.team == 1:
                agent.state.p_pos = np.random.uniform(-.9, .9, world.dim_p)
            else:
                raise f"Undefined start location for agent in team {agent.team}"

            agent.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        if agent.team == 0:
            return -10 * agent.state.biting
        elif agent.team == 1:
            return 10 * agent.state.biting
        else:
            raise "Undefined reward for team %d" % agent.team

    def observation(self, agent, world):
        agents = np.array(world.agents)
        s = (2 + 2) * len(agents)
        obs = np.zeros((s,))
        i = 0

        # sort agents by distance from agent
        pos = np.array([a.state.p_pos for a in agents])
        delta_pos = pos - agent.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos), axis=1))
        agents = agents[np.argsort(dist)]

        # sort again by team
        humans = [a for a in agents if a.team == 0]
        zombies = [a for a in agents if a.team == 1]
        agents = humans + zombies

        for other in agents:
            if other is agent:
                # absolute position of agent
                obs[i:i+2] = agent.state.p_pos; i += 2
                obs[i:i+2] = agent.state.p_vel; i += 2
            else:
                # relative position of all other agents
                obs[i:i+2] = other.state.p_pos - agent.state.p_pos; i += 2
                obs[i:i+2] = other.state.p_vel - agent.state.p_vel; i += 2
        assert i == s

        return obs

