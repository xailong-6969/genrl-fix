from genrl_swarm.examples.gsm8k.reward_utils import format_reward, correctness_reward, validity_reward, extract_responses, parse_game_state

#TODO(gab): Double check these regex more carefully+add tests
    
class Stage0Rewards:
    def __init__(self):
        self.stage = 0
        self.reward_fn = self.cumulative_reward

    def cumulative_reward(self, completions, answer):
        if completions is None or not completions or not isinstance(completions, list):
            return [0.0]
        if answer is None or not answer:
            return [0.0] * len(completions)
        
        formatting = format_reward(completions, pattern=r"\nAnswer: \d+")
        correctness = correctness_reward(completions, answer, pattern=r'Answer: .*?([\d,]+(?:\.\d+)?)')
        #TODO: Come back and add delayed rewards 

        cumulative = [sum(tup) for tup in zip(formatting, correctness)]
        return cumulative
    
    def __call__(self, game_state):
        completions, answers = parse_game_state(game_state, self.stage)
        rewards = {} #Key per agent
        for agent in completions:
            rewards[agent] = [] #Will store a list per batch item
            for batch_idx, _ in enumerate(completions[agent]):
                batch_rewards = []
                for node_idx, _ in enumerate(completions[agent][batch_idx]):
                    batch_rewards.append(self.reward_fn(completions[agent][batch_idx][node_idx], answers[agent][batch_idx][node_idx]))
                rewards[agent].append(batch_rewards)
        return rewards
    
class Stage1Rewards:
    def __init__(self):
        self.stage = 1
        self.reward_fn = self.cumulative_reward

    def cumulative_reward(self, completions, choices):
        if completions is None or not completions or not isinstance(completions, list):
            return [0.0]
        if choices is None or not choices:
            return [0.0] * len(completions)
        
        formatting = format_reward(completions, pattern=r"\nChoice: Student #\d+")
        validity = validity_reward(completions, valid=[str(idx) for idx, _ in enumerate(choices)], pattern=r'Choice:.*?Student #([\d]+(?:\.\d+)?)')
        ##TODO: Come back and add the correctness reward for choosing majority answer + reward for choosing a mathematically correct answer
        # #Find majority choice and give reward if choice aligns
        # majority_choice = Counter([extract_responses(c) for c in choices]).most_common(1)[0][0]
        # correctness = correctness_reward(completions, majority_choice, pattern=r'Choice:.*?Student #([\d]+(?:\.\d+)?)')

        cumulative = [sum(tup) for tup in zip(formatting, validity)]
        return cumulative
    
    def __call__(self, game_state):
        completions, valid = parse_game_state(game_state, self.stage)
        rewards = {} #Key per agent
        for agent in completions:
            rewards[agent] = [] #Will store a list per batch item
            for batch_idx, _ in enumerate(completions[agent]):
                batch_rewards = []
                for node_idx, _ in enumerate(completions[agent][batch_idx]):
                    batch_rewards.append(self.reward_fn(completions[agent][batch_idx][node_idx], valid[agent][batch_idx][node_idx]))
                rewards[agent].append(batch_rewards)
        return rewards
    
class Stage2Rewards:
    def __init__(self):
        self.stage = 2
        self.reward_fn = self.cumulative_reward

    def cumulative_reward(self, completions, answer):
        if completions is None or not completions or not isinstance(completions, list):
            return [0.0]
        if answer is None or not answer:
            return [0.0] * len(completions)
        
        formatting = format_reward(completions, pattern=r"\nAnswer: \d+")
        correctness = correctness_reward(completions, answer, pattern=r'Answer: .*?([\d,]+(?:\.\d+)?)')
        #TODO: Come back and add improvement rewards compared to stage 0 

        cumulative = [sum(tup) for tup in zip(formatting, correctness)]
        return cumulative
    
    def __call__(self, game_state):
        completions, answers = parse_game_state(game_state, self.stage)
        rewards = {} #Key per agent
        for agent in completions:
            rewards[agent] = [] #Will store a list per batch item
            for batch_idx, _ in enumerate(completions[agent]):
                batch_rewards = []
                for node_idx, _ in enumerate(completions[agent][batch_idx]):
                    batch_rewards.append(self.reward_fn(completions[agent][batch_idx][node_idx], answers[agent][batch_idx][node_idx]))
                rewards[agent].append(batch_rewards)
        return rewards