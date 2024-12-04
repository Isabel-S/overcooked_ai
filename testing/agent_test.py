import unittest

import numpy as np
import random

from overcooked_ai_py.agents.agent import (
    AgentPair,
    FixedPlanAgent,
    GreedyHumanModel,
    RandomAgent,
    SampleAgent,
    QLearningAgent
)
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import (
    ObjectState,
    OvercookedGridworld,
    OvercookedState,
    PlayerState,
)
from overcooked_ai_py.planning.planners import (
    NO_COUNTERS_PARAMS,
    MediumLevelActionManager,
)

np.random.seed(42)

n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState

force_compute_large = False
force_compute = True
DISPLAY = False

# Map actions to descriptive labels
ACTION_MAP = {
    (0, -1): "MOVE_UP",
    (0, 1): "MOVE_DOWN",
    (-1, 0): "MOVE_LEFT",
    (1, 0): "MOVE_RIGHT",
    (0, 0): "STAY",
    "INTERACT": "INTERACT"
}

simple_mdp = OvercookedGridworld.from_layout_name("cramped_room")
large_mdp = OvercookedGridworld.from_layout_name("corridor")




class TestAgentEvaluator(unittest.TestCase):
    def setUp(self):
        self.agent_eval = AgentEvaluator.from_layout_name(
            {"layout_name": "cramped_room"}, {"horizon": 100}
        )

    def test_human_model_pair(self):
        trajs = self.agent_eval.evaluate_human_model_pair()
        try:
            AgentEvaluator.check_trajectories(trajs, verbose=False)
        except AssertionError as e:
            self.fail(
                "Trajectories were not returned in standard format:\n{}".format(
                    e
                )
            )

    def test_rollouts(self):
        ap = AgentPair(RandomAgent(), RandomAgent())
        trajs = self.agent_eval.evaluate_agent_pair(ap, num_games=5)
        try:
            AgentEvaluator.check_trajectories(trajs, verbose=False)
        except AssertionError as e:
            self.fail(
                "Trajectories were not returned in standard format:\n{}".format(
                    e
                )
            )

    def test_mlam_computation(self):
        try:
            self.agent_eval.env.mlam
        except Exception as e:
            self.fail(
                "Failed to compute MediumLevelActionManager:\n{}".format(e)
            )


class TestBasicAgents(unittest.TestCase):
    def setUp(self):
        self.mlam_large = MediumLevelActionManager.from_pickle_or_compute(
            large_mdp, NO_COUNTERS_PARAMS, force_compute=force_compute_large
        )
    
    
        
    def train_two_agents_simultaneously(self, agent_0, agent_1, env, num_episodes=500, max_steps_per_episode=50, start_state=None):
        for episode in range(num_episodes):
            print(f"Starting Episode {episode + 1}/{num_episodes}...")

            # Reset environment to fixed start state or default
            if start_state:
                state = start_state
            else:
                env.reset()
            

            for step in range(max_steps_per_episode):
                print(f"  Step {step + 1}/{max_steps_per_episode}...")

                # Get actions for both agents
                action_0, _ = agent_0.action(state)
                action_1, _ = agent_1.action(state)

                # Execute actions in the environment
                joint_action = [action_0, action_1]
                next_state, reward, done, _ = env.step(joint_action)

                # Debug: Print state, actions, and rewards
                print(f"    State: {state}")
                print(f"    Actions: {joint_action}")
                print(f"    Reward: {reward} (Type: {type(reward)})")
                print(f"    Next State: {next_state}")
                print(f"    Done: {done}")

                # Update agents using the received reward
                agent_0.update(state, action_0, reward, next_state)
                agent_1.update(state, action_1, reward, next_state)

                # Break if the environment signals termination
                if done:
                    print(f"    Episode terminated early at step {step + 1}.")
                    break

                # Move to the next state
                state = next_state

            print(f"Episode {episode + 1} complete.\n")

    def test_two_qlearning_agents_cramped_room_with_debugging(self):
        """
        Test two QLearningAgents on the 'cramped_room' layout, including debugging statements.
        """
        layout = "cramped_room"
        mdp = OvercookedGridworld.from_layout_name(layout)
        env = OvercookedEnv.from_mdp(mdp, horizon=50)

        # Initialize QLearningAgents
        agent_0 = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.1, action_space=Action.ALL_ACTIONS)
        agent_1 = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.1, action_space=Action.ALL_ACTIONS)

        # Train both agents to populate their Q-tables
        fixed_start_state = mdp.get_standard_start_state()
        self.train_two_agents_simultaneously(
        agent_0=agent_0,
        agent_1=agent_1,
        env=env,
        num_episodes=500,
        max_steps_per_episode=50,
        start_state=fixed_start_state
        )

        agent_pair = AgentPair(agent_0, agent_1)

        # Run the agents multiple times to validate Q-table persistence
        for run_idx in range(3):
            print(f"Run {run_idx + 1} starting...")
            state = env.reset()
            env.state = fixed_start_state  # Reset to the fixed start state for testing
            trajectory, time_taken, _, _ = env.run_agents(
                agent_pair, include_final_state=True, display=False
            )
            assert trajectory is not None, "Trajectory should not be None"
            assert time_taken <= 50, "Time taken should not exceed horizon"
            assert len(agent_0.q_table) > 0, "Agent 0's Q-table should persist after reset."
            assert len(agent_1.q_table) > 0, "Agent 1's Q-table should persist after reset."

            # Debug: Check Q-tables
            print(f"Agent 0 Q-Table Size: {len(agent_0.q_table)}")
            print(f"Agent 1 Q-Table Size: {len(agent_1.q_table)}")

            # Print the trajectory details with debugging
            for timestep, step in enumerate(trajectory):
                state, actions, reward, done, metadata = step

                # Map actions to descriptive labels
                ACTION_MAP = {
                    (0, -1): "MOVE_UP",
                    (0, 1): "MOVE_DOWN",
                    (-1, 0): "MOVE_LEFT",
                    (1, 0): "MOVE_RIGHT",
                    (0, 0): "STAY",
                    "INTERACT": "INTERACT"
                }

                action_labels = tuple(ACTION_MAP.get(a, a) for a in actions)

                sparse_rewards = metadata.get("sparse_r_by_agent", [0, 0]) if metadata else [0, 0]
                shaped_rewards = metadata.get("shaped_r_by_agent", [0, 0]) if metadata else [0, 0]
                total_rewards = sparse_rewards[0] + sparse_rewards[1]

                print(f"Timestep {timestep}:")
                print(f"  State: {state}")
                print(f"  Hashed State: {hash(state)}")
                print(f"  Actions: {action_labels}")
                print(f"  Reward: {reward}")
                print(f"  Sparse Rewards by Agent: {sparse_rewards}")
                print(f"  Shaped Rewards by Agent: {shaped_rewards}")
                print(f"  Total Rewards: {total_rewards}")

                # Debug: Check if no rewards are being generated
                if sparse_rewards == [0, 0] and shaped_rewards == [0, 0]:
                    print(f"Warning: No rewards generated at Timestep {timestep}")
                
                # Debug: Check if agents are stuck (e.g., repetitive states)
                if timestep > 0 and trajectory[timestep - 1][0] == state:
                    print(f"Warning: Agents may be stuck at Timestep {timestep}, State: {state}")

            print(f"Run {run_idx + 1} completed.\n")

    # def test_fixed_plan_agents(self):
    #     a0 = FixedPlanAgent([s, e, n, w])
    #     a1 = FixedPlanAgent([s, w, n, e])
    #     agent_pair = AgentPair(a0, a1)
    #     env = OvercookedEnv.from_mdp(large_mdp, horizon=10)
    #     trajectory, time_taken, _, _ = env.run_agents(
    #         agent_pair, include_final_state=True, display=DISPLAY
    #     )
    #     end_state = trajectory[-1][0]
    #     self.assertEqual(time_taken, 10)
    #     self.assertEqual(
    #         env.mdp.get_standard_start_state().player_positions,
    #         end_state.player_positions,
    #     )

    # def test_two_greedy_human_open_map(self):
    #     scenario_2_mdp = OvercookedGridworld.from_layout_name("scenario2")
    #     mlam = MediumLevelActionManager.from_pickle_or_compute(
    #         scenario_2_mdp, NO_COUNTERS_PARAMS, force_compute=force_compute
    #     )
    #     a0 = GreedyHumanModel(mlam)
    #     a1 = GreedyHumanModel(mlam)
    #     agent_pair = AgentPair(a0, a1)
    #     start_state = OvercookedState(
    #         [P((8, 1), s), P((1, 1), s)],
    #         {},
    #         all_orders=scenario_2_mdp.start_all_orders,
    #     )
    #     env = OvercookedEnv.from_mdp(
    #         scenario_2_mdp, start_state_fn=lambda: start_state, horizon=100
    #     )
    #     trajectory, time_taken, _, _ = env.run_agents(
    #         agent_pair, include_final_state=True, display=DISPLAY
    #     )

    #     # output trajectory
    #     print("Greedy Human Model Open Map Trajectory")
    #     for timestep, step in enumerate(trajectory):
    #         state, actions, reward, done, metadata = step

    #         action_labels = tuple(ACTION_MAP.get(a, a) for a in actions)

    #         sparse_rewards = metadata.get("sparse_r_by_agent", [0, 0]) if metadata else [0, 0]
    #         shaped_rewards = metadata.get("shaped_r_by_agent", [0, 0]) if metadata else [0, 0]
    #         print(f"Timestep {timestep}:")
    #         print(f"  State: {state}")
    #         print(f"  Actions: {action_labels}")
    #         print(f"  Sparse Rewards by Agent: {sparse_rewards}")
    #         print(f"  Shaped Rewards by Agent: {shaped_rewards}")
    #         print("--------")
    #     print("Total rewards: ", sum([step[2] for step in trajectory]))

    
    # def test_two_greedy_human_compact(self):
    #     layout = "cramped_room"
    #     mdp = OvercookedGridworld.from_layout_name(layout)
    #     mlam = MediumLevelActionManager.from_pickle_or_compute(
    #         mdp, NO_COUNTERS_PARAMS, force_compute=True
    #     )

    #     a0 = GreedyHumanModel(mlam)
    #     a1 = GreedyHumanModel(mlam)
    #     agent_pair = AgentPair(a0, a1)
    #     start_state = mdp.get_standard_start_state()

    #     env = OvercookedEnv.from_mdp(
    #         mdp, start_state_fn=lambda: start_state, horizon=100
    #     )
    #     trajectory, time_taken, _, _ = env.run_agents(
    #         agent_pair, include_final_state=True, display=False
    #     )

    #     print("Greedy Human Model Compact Room Trajectory:")
    #     for timestep, step in enumerate(trajectory):
    #         state, actions, reward, done, metadata = step

    #         # Map actions to descriptive labels
    #         ACTION_MAP = {
    #             (0, -1): "MOVE_UP",
    #             (0, 1): "MOVE_DOWN",
    #             (-1, 0): "MOVE_LEFT",
    #             (1, 0): "MOVE_RIGHT",
    #             (0, 0): "STAY",
    #             "INTERACT": "INTERACT"
    #         }

    #         action_labels = tuple(ACTION_MAP.get(a, a) for a in actions)

    #         sparse_rewards = metadata.get("sparse_r_by_agent", [0, 0]) if metadata else [0, 0]
    #         shaped_rewards = metadata.get("shaped_r_by_agent", [0, 0]) if metadata else [0, 0]
    #         print(f"Timestep {timestep}:")
    #         print(f"  State: {state}")
    #         print(f"  Actions: {action_labels}")
    #         print(f"  Rewards: {reward}")
    #         print(f"  Sparse Rewards by Agent: {sparse_rewards}")
    #         print(f"  Shaped Rewards by Agent: {shaped_rewards}")
    #         print("--------")
    #     print("Total rewards: ", sum([step[2] for step in trajectory]))


    # def test_sample_agent(self):
    #     agent = SampleAgent(
    #         [RandomAgent(all_actions=False), RandomAgent(all_actions=True)]
    #     )
    #     probs = agent.action(None)[1]["action_probs"]
    #     expected_probs = np.array(
    #         [
    #             0.18333333,
    #             0.18333333,
    #             0.18333333,
    #             0.18333333,
    #             0.18333333,
    #             0.08333333,
    #         ]
    #     )
    #     self.assertTrue(np.allclose(probs, expected_probs))


class TestAgentEvaluatorStatic(unittest.TestCase):
    layout_name_lst = [
        "asymmetric_advantages",
        "asymmetric_advantages_tomato",
        "bonus_order_test",
        "bottleneck",
        "centre_objects",
        "centre_pots",
        "corridor",
        "forced_coordination_tomato",
        "unident",
        "marshmallow_experiment",
        "marshmallow_experiment_coordination",
        "you_shall_not_pass",
    ]

    def test_from_mdp(self):
        for layout_name in self.layout_name_lst:
            orignal_mdp = OvercookedGridworld.from_layout_name(layout_name)
            ae = AgentEvaluator.from_mdp(
                mdp=orignal_mdp, env_params={"horizon": 400}
            )
            ae_mdp = ae.env.mdp
            self.assertEqual(
                orignal_mdp,
                ae_mdp,
                "mdp with name "
                + layout_name
                + " experienced an inconsistency",
            )

    def test_from_mdp_params_layout(self):
        for layout_name in self.layout_name_lst:
            orignal_mdp = OvercookedGridworld.from_layout_name(layout_name)
            ae = AgentEvaluator.from_layout_name(
                mdp_params={"layout_name": layout_name},
                env_params={"horizon": 400},
            )
            ae_mdp = ae.env.mdp
            self.assertEqual(
                orignal_mdp,
                ae_mdp,
                "mdp with name "
                + layout_name
                + " experienced an inconsistency",
            )

    mdp_gen_params_1 = {
        "inner_shape": (10, 7),
        "prop_empty": 0.95,
        "prop_feats": 0.1,
        "start_all_orders": [{"ingredients": ["onion", "onion", "onion"]}],
        "display": False,
    }

    mdp_gen_params_2 = {
        "inner_shape": (10, 7),
        "prop_empty": 0.7,
        "prop_feats": 0.5,
        "start_all_orders": [{"ingredients": ["onion", "onion", "onion"]}],
        "display": False,
    }

    mdp_gen_params_3 = {
        "inner_shape": (10, 7),
        "prop_empty": 0.5,
        "prop_feats": 0.4,
        "start_all_orders": [{"ingredients": ["onion", "onion", "onion"]}],
        "display": False,
    }

    mdp_gen_params_lst = [mdp_gen_params_1, mdp_gen_params_2, mdp_gen_params_3]

    outer_shape = (10, 7)

    def test_from_mdp_params_variable_across(self):
        for mdp_gen_params in self.mdp_gen_params_lst:
            ae0 = AgentEvaluator.from_mdp_params_infinite(
                mdp_params=mdp_gen_params,
                env_params={"horizon": 400, "num_mdp": np.inf},
                outer_shape=self.outer_shape,
            )
            ae1 = AgentEvaluator.from_mdp_params_infinite(
                mdp_params=mdp_gen_params,
                env_params={"horizon": 400, "num_mdp": np.inf},
                outer_shape=self.outer_shape,
            )
            self.assertFalse(
                ae0.env.mdp == ae1.env.mdp,
                "2 randomly generated layouts across 2 evaluators are the same, which is wrong",
            )

    def test_from_mdp_params_variable_infinite(self):
        for mdp_gen_params in self.mdp_gen_params_lst:
            ae = AgentEvaluator.from_mdp_params_infinite(
                mdp_params=mdp_gen_params,
                env_params={"horizon": 400, "num_mdp": np.inf},
                outer_shape=self.outer_shape,
            )
            mdp_0 = ae.env.mdp.copy()
            for _ in range(5):
                ae.env.reset(regen_mdp=True)
                mdp_1 = ae.env.mdp
                self.assertFalse(
                    mdp_0 == mdp_1,
                    "with infinite layout generator and regen_mdp=True, the 2 layouts should not be the same",
                )

    def test_from_mdp_params_variable_infinite_no_regen(self):
        for mdp_gen_params in self.mdp_gen_params_lst:
            ae = AgentEvaluator.from_mdp_params_infinite(
                mdp_params=mdp_gen_params,
                env_params={"horizon": 400, "num_mdp": np.inf},
                outer_shape=self.outer_shape,
            )
            mdp_0 = ae.env.mdp.copy()
            for _ in range(5):
                ae.env.reset(regen_mdp=False)
                mdp_1 = ae.env.mdp
                self.assertTrue(
                    mdp_0 == mdp_1,
                    "with infinite layout generator and regen_mdp=False, the 2 layouts should be the same",
                )

    def test_from_mdp_params_variable_infinite_specified(self):
        for mdp_gen_params in self.mdp_gen_params_lst:
            ae = AgentEvaluator.from_mdp_params_infinite(
                mdp_params=mdp_gen_params,
                env_params={"horizon": 400, "num_mdp": np.inf},
                outer_shape=self.outer_shape,
            )
            mdp_0 = ae.env.mdp.copy()
            for _ in range(5):
                ae.env.reset(regen_mdp=True)
                mdp_1 = ae.env.mdp
                self.assertFalse(
                    mdp_0 == mdp_1,
                    "with infinite layout generator and regen_mdp=True, the 2 layouts should not be the same",
                )

    def test_from_mdp_params_variable_finite(self):
        for mdp_gen_params in self.mdp_gen_params_lst:
            ae = AgentEvaluator.from_mdp_params_finite(
                mdp_params=mdp_gen_params,
                env_params={"horizon": 400, "num_mdp": 2},
                outer_shape=self.outer_shape,
            )
            mdp_0 = ae.env.mdp.copy()
            seen = [mdp_0]
            for _ in range(20):
                ae.env.reset(regen_mdp=True)
                mdp_i = ae.env.mdp
                if len(seen) == 1:
                    if mdp_i != seen[0]:
                        seen.append(mdp_i.copy())
                elif len(seen) == 2:
                    mdp_0, mdp_1 = seen
                    self.assertTrue(
                        (mdp_i == mdp_0 or mdp_i == mdp_1),
                        "more than 2 mdp was created, the function failed to perform",
                    )
                else:
                    self.assertTrue(
                        False, "theoretically unreachable statement"
                    )

    layout_name_short_lst = [
        "cramped_room",
        "cramped_room_tomato",
        "simple_o",
        "simple_tomato",
        "simple_o_t",
    ]
    biased = [0.1, 0.15, 0.2, 0.25, 0.3]
    num_reset = 200000

    def test_from_mdp_lst_default(self):
        mdp_lst = [
            OvercookedGridworld.from_layout_name(name)
            for name in self.layout_name_short_lst
        ]
        ae = AgentEvaluator.from_mdp_lst(
            mdp_lst=mdp_lst, env_params={"horizon": 400}
        )
        counts = {}

        for _ in range(self.num_reset):
            ae.env.reset(regen_mdp=True)
            if ae.env.mdp.layout_name in counts:
                counts[ae.env.mdp.layout_name] += 1
            else:
                counts[ae.env.mdp.layout_name] = 1

        for k, v in counts.items():
            self.assertAlmostEqual(
                0.2, v / self.num_reset, 2, "more than 2 places off for " + k
            )

    def test_from_mdp_lst_uniform(self):
        mdp_lst = [
            OvercookedGridworld.from_layout_name(name)
            for name in self.layout_name_short_lst
        ]
        ae = AgentEvaluator.from_mdp_lst(
            mdp_lst=mdp_lst,
            env_params={"horizon": 400},
            sampling_freq=[0.2, 0.2, 0.2, 0.2, 0.2],
        )
        counts = {}

        for _ in range(self.num_reset):
            ae.env.reset(regen_mdp=True)
            if ae.env.mdp.layout_name in counts:
                counts[ae.env.mdp.layout_name] += 1
            else:
                counts[ae.env.mdp.layout_name] = 1

        for k, v in counts.items():
            self.assertAlmostEqual(
                0.2, v / self.num_reset, 2, "more than 2 places off for " + k
            )

    def test_from_mdp_lst_biased(self):
        mdp_lst = [
            OvercookedGridworld.from_layout_name(name)
            for name in self.layout_name_short_lst
        ]
        ae = AgentEvaluator.from_mdp_lst(
            mdp_lst=mdp_lst,
            env_params={"horizon": 400},
            sampling_freq=self.biased,
        )
        counts = {}

        for _ in range(self.num_reset):
            ae.env.reset(regen_mdp=True)
            if ae.env.mdp.layout_name in counts:
                counts[ae.env.mdp.layout_name] += 1
            else:
                counts[ae.env.mdp.layout_name] = 1

        # construct the ground truth
        gt = {
            self.layout_name_short_lst[i]: self.biased[i]
            for i in range(len(self.layout_name_short_lst))
        }

        for k, v in counts.items():
            self.assertAlmostEqual(
                gt[k], v / self.num_reset, 2, "more than 2 places off for " + k
            )


if __name__ == "__main__":
    unittest.main()
