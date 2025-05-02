import abc

from genrl_swarm.state import GameState
from genrl_swarm.rewards import RewardManager
from genrl_swarm.trainer import TrainerModule
from genrl_swarm.data import DataManager
from genrl_swarm.roles import RoleManager #TODO: Add this in when ready. Either will need to pass an arg to trainer OR control which trainer from some set of trainers to use OR ...?

class GameManager(abc.ABC): #TODO(gabe): Make this use enum
    def __init__(self, 
                 game_state:GameState, 
                 reward_manager:RewardManager, 
                 trainer:TrainerModule, 
                 data_manager:DataManager, 
                 role_manager:RoleManager = None,
                 run_mode:str = "Train"
                 ):
        """Initialization method that stores the various managers needed to orchestrate this game"""
        self.state = game_state
        self.rewards = reward_manager
        self.trainer = trainer
        self.data_manager = data_manager
        self.roles = role_manager
        self.mode = run_mode.lower()
    
    @abc.abstractmethod
    def end_of_game(self, **kwargs) -> bool:
        """
        Defines conditions for the game to end and no more rounds/stage should begin. 
        Return True if conditions imply game should end, else False
        """
        pass

    @abc.abstractmethod
    def end_of_round(self, **kwargs) -> bool:
        """
        Defines conditions for end of a round AND no more stages/"turns" should being for this round AND the game state should be reset for stage 0 of your game. 
        Return True if conditions imply game should end and no new round/stage should begin, else False
        """
        pass

    @abc.abstractmethod
    def prune_function(self, **kwargs) -> None:
        """
        Prunes examples held in the game state to reduce the size (e.g., in large swarms) and/or keep only examples which are desired for a model's learning.
        """
        pass

    def run_game_stage(self):
        ####################
        #TODO(group): Decide whether we want state to give inputs to trainer in here OR if we want a callable to be passed to trainer's init that will fetch the input per stage from the data in state
        # inputs = self.state.get_latest() # Fetches the current batch of prepared model inputs
        # outputs = self.trainer.generate(inputs) # Generates a rollout 
        ####################
        outputs = self.trainer.generate() # Generates a rollout 
        self.state.append_generation(outputs) # Adds the freshly generated rollout to the game state for pruning, etc. 

    def run_game_round(self):
        # Loop through stages until end of round is hit
        while not self.end_of_round():
            self.run_game_stage() # Generates rollout and updates the game state
            self.state.advance_stage() # Prepare for next stage
        self.rewards.update_rewards(self.state) # Compute reward functions now that we have all the data needed for this round
        if self.mode in ['train', 'train_and_evaluate']:
            self.trainer.train(self.state, self.rewards)
        if self.mode in ['evaluate', 'train_and_evaluate']:
            self.trainer.evaluate(self.data_manager, self.rewards)
        train_batch = self.data_manager.train_batch() #TODO(johnny): Add in a param/option so that we can (globally/consistently across agents) subsample as a function of round
        self.state.advance_round(train_batch) # Resets the game state appropriately, stages the next round, and increments round/stage counters appropriatelly

    def run_game(self):
        # Loop through rounds until end of the game is hit
        while not self.end_of_game():
            self.run_game_round() # Loops through stages until end of round signal is received


class BaseGameManager(GameManager):
    """
    Default GameManager with some basic functionality baked-in.
    Will end the game when max_rounds is reached, end a round when max_stage is reached, and prune according to top-k rewards from previous stage.
    """
    def __init__(self,
                 max_stage:int,
                 max_round:int,
                 prune_K:int,
                 game_state:GameState, 
                 reward_manager:RewardManager, 
                 trainer:TrainerModule, 
                 data_manager:DataManager, 
                 role_manager:RoleManager = None,
                 run_mode:str = "Train"
                 ):
        """Init a GameManager which ends the game when max_rounds is reached, ends stage when max_stage is reached, and prunes according to top-k rewards"""
        self.max_stage = max_stage
        self.max_round = max_round
        self.prune_K = prune_K
        kwargs = {"game_state":game_state, 
                  "reward_manager":reward_manager, 
                  "trainer":trainer, 
                  "data_manager":data_manager, 
                  "role_manager":role_manager, 
                  "run_mode":run_mode
                  }
        super.__init__(**kwargs)

    def end_of_game(self) -> bool:
        if self.state.round > self.max_round:
            return True
        else:
            return False
    
    def end_of_round(self) -> bool:
        if self.state.stage > self.max_stage:
            return True
        else:
            return False

    def prune_function(self) -> None:
        #TODO(gabe OR jari): Pass self.prune_k to a default top-k pruner function. Note that pruner must be consistent across all agents in the swarm by default!
        pass