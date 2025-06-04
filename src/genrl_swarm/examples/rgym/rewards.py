from genrl_swarm.examples.rgym.reward_utils import *

class RGRewards:
    def __init__(self):
        self.stage = 0
        self.reward_fn = self.cumulative_reward
    
    def cumulative_reward(self, completions, answer, metadata):
        if completions is None or not completions or not isinstance(completions, list):
            return [0.0]
        if answer is None or not answer:
            return [0.0] * len(completions)
        
        formatting = format_reward(completions)
        correctness = accuracy_reward(completions, answer, metadata)

        cumulative = [sum(tup) for tup in zip(formatting, correctness)]
        return cumulative
    
    def __call__(self, game_state):
        completions, answers, metadata = parse_game_state(game_state, self.stage)
        rewards = {} #Key per agent
        for agent in completions:
            rewards[agent] = {} #Will store a list per batch item
            for batch_id in completions[agent]:
                rewards[agent][batch_id] = []
                for node_idx, _ in enumerate(completions[agent][batch_id]):
                    rewards[agent][batch_id].append(self.reward_fn(completions[agent][batch_id][node_idx], answers[agent][batch_id][node_idx], metadata[agent][batch_id][node_idx]))
        return rewards