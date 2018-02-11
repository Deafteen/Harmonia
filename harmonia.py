# python -m pysc2.bin.agent --map Simple64 --agent pysc2.agents.Harmonia.harmonia.Harmonia --agent_race T

# python -m pysc2.bin.agent --map Simple64 --agent pysc2.agents.Harmonia.harmonia.SmartHarmonia --agent_race T --max_agent_steps 0 --norender

import random
import math

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import time

# Functions
_NO_OP = actions.FUNCTIONS.no_op.id
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

# Unit IDs
_TERRAN_COMMANDCENTER = 18
_TERRAN_BARRACKS = 21
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19

# Parameters
_PLAYER_SELF = 1
_NOT_QUEUED = [0]
_QUEUED = [1]
_SUPPLY_USED = 3
_SUPPLY_MAX = 4




class Harmonia(base_agent.BaseAgent):
    # Properties
    base_top_left = None
    supply_depot_built = False
    scv_selected = False
    barracks_built = False
    barracks_selected = False
    barracks_rallied = False
    army_selected = False
    army_rallied = False


    def step(self, obs):
        super(Harmonia, self).step(obs)

        #time.sleep(0.5)

        if self.base_top_left is None:
            player_y, player_x = (obs.observation["minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = player_y.mean() <= 31


        if not self.supply_depot_built:
            if not self.scv_selected:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

                target = [unit_x[0], unit_y[0]]

                self.scv_selected = True

                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            elif _BUILD_SUPPLYDEPOT in obs.observation["available_actions"]:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                target = self.transformLocation(int(unit_x.mean()), 0, int(unit_y.mean()), 20)

                self.supply_depot_built = True

                return actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_NOT_QUEUED, target])

        if not self.barracks_built:
            if _BUILD_BARRACKS in obs.observation["available_actions"]:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 0)

                self.barracks_built = True

                return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

        elif not self.barracks_rallied:
            if not self.barracks_selected:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()

                if unit_y.any():
                    target = [int(unit_x.mean()), int(unit_y.mean())]

                    self.barracks_selected = True

                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            else:
                self.barracks_rallied = True

                if self.base_top_left:
                    return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 21]])

                return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 46]])

        elif obs.observation["player"][_SUPPLY_USED] < obs.observation["player"][_SUPPLY_MAX] and _TRAIN_MARINE in \
                obs.observation["available_actions"]:
            return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])


        elif not self.army_rallied:
            if not self.army_selected:
                if _SELECT_ARMY in obs.observation["available_actions"]:
                    self.army_selected = True
                    self.barracks_selected = False

                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

            elif _ATTACK_MINIMAP in obs.observation["available_actions"]:
                #self.army_rallied = True
                self.army_selected = False

                if self.base_top_left:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [39, 45]])
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [21, 24]])



        
        return actions.FunctionCall(_NOOP, [])


    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]


class SmartHarmonia(base_agent.BaseAgent):
    base_top_left = None

    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def step(self, obs):
        super(SmartHarmonia, self).step(obs)

        player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        return actions.FunctionCall(_NO_OP, [])

    # Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
    class QLearningTable:
        def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
            self.actions = actions
            self.lr = learning_rate
            self.gamma = reward_decay
            self.epsilon = e_greedy
            self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

        def choose_action(self, observation):
            self.check_state_exist(observation)

            if np.random.uniform() < self.epsilon:
                # choose best action
                state_action = self.q_table.ix[observation, :]

                # some actions have the same value
                state_action = state_action.reindex(np.random.permutation(state_action.index))

                action = state_action.idxmax()
            else:
                # choose random action
                action = np.random.choice(self.actions)

            return action

        def learn(self, s, a, r, s_):
            self.check_state_exist(s_)
            self.check_state_exist(s)

            q_predict = self.q_table.ix[s, a]
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()

            # update
            self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

        def check_state_exist(self, state):
            if state not in self.q_table.index:
                # append new state to q table
                self.q_table = self.q_table.append(
                    pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))
