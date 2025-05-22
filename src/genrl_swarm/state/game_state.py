from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Callable
from genrl_swarm.state.game_tree import GameTree, DefaultGameTree

@dataclass
class GameState:
    round: int
    stage: int
    swarm_size: int = 1 # Number of agents

    trees: Dict[Any, Dict[int, Callable]] | None = None # [Agent][Batch][GameTree]    
    game_tree_factory: GameTree = DefaultGameTree #GameTree data structure to use for building tree
    
    def __post_init__(self) -> None:
        if self.round < 0:
            self.round = 0
        if self.stage < 0:
            self.stage = 0
        self.world_state_pruners = {"environment_pruner": None, "opponent_pruner": None, "personal_pruner": None}
        self.tree_branching_functions = {"terminal_node_decision_function": None, "stage_inheritance_function": None}

    
    def _init_game(self, 
                   round_data: List[Tuple[Any]], 
                   world_state_pruners: Dict[str, Callable | None] = None,
                   game_tree_brancher: Dict[str, Callable | None] = None
                   ) -> None:
        """
        For each agent this builds a game tree with only a root for each datum in the round_data batch
        
        Inputs:
            Tuple[Any]: Tuple of world state needed for defining an agent's input for this node. Has form (environment_states, opponent_states, personal_states)
        """
        if world_state_pruners is not None:
            self.world_state_pruners = world_state_pruners #Game-specific pruner functions for passing to game trees. Defines what state details are relevant to a stage + should be broacast. NOTE: Using None here means no pruning of that type of state!
        if game_tree_brancher is not None:
            self.tree_branching_functions = game_tree_brancher #Game-specific functions for passing to game trees to define when they should consider a node "terminal" + when/how a node should be branched to produce children
        
        self.batch_size = len(round_data)
        
        self.trees = {}
        for agent in range(self.swarm_size):
            self.trees[agent] = {}
            for batch_idx in range(self.batch_size):
                self.trees[agent][batch_idx] = self.game_tree_factory(round_data[batch_idx]) #NOTE: Assumes round data is a list containting tuples with indices for environment, opponent, and personal states respectively

    def advance_round(self, round_data: List[Tuple[Any]]) -> None:
        """Increments round, restarts stage count, and initializes game tree on the new round's data"""
        self.round += 1
        self.stage = 0
        self._init_game(round_data)
        
    def get_stage_state(self, stage_num: int) -> Dict[Any, List[List[List[Any]]]]:
        """
        Returns required state information for preparing a data batch from the desired stage
        
        Returns:
            Dict[List[List[Tuple[Any]]]]: Dict keyed on agents containing a list for each batch item that contains a list for each node for the desired stage. 
                                          The list per node in the desired stage contains tuples with state information required for preparing the corresponding agent's data batch for the desired stage. 
                                          Tuples have the form (environment_states, opponent_states, personal_states)
        """
        if self.trees and len(self.trees) > 0:
            agents = {}
            for agent_idx in range(self.swarm_size):
                agents[agent_idx] = []
                for batch_idx in range(self.batch_size):
                    batch_states = []
                    batch_nodes = self.trees[agent_idx][batch_idx][stage_num]
                    for node_idx in range(len(batch_nodes)):
                        world_state = [batch_nodes[node_idx]['environment_states'], 
                                       batch_nodes[node_idx]['opponent_states'], 
                                       batch_nodes[node_idx]['personal_states']
                                       ]
                        batch_states.append(world_state)
                    agents[agent_idx].append(batch_states)
            return agents # [Agents][Batch][Node Idx in Stage][World State]
        else:
            raise RuntimeError("Trying to get game state information, but game trees are not defined.")
        
    def get_stage_actions(self, stage_num: int) -> Dict[Any, List[List[Any]]]:
        """
        Returns actions stored in the nodes of a stage
        NOTE: Behaves similarly to get_stage_state, but loses a dimension since actions are a singleton object within nodes. That said, they can be any type depending on game specifics.

        Returns:
            Dict[List[List[Any]]]: Dict keyed on agents containing a list for each batch item that contains a list for each node for the desired stage. 
                                   The list per node in the desired stage contains actions for said node.
        """
        if self.trees and len(self.trees) > 0:
            agents = {}
            for agent_idx in range(self.swarm_size):
                agents[agent_idx] = []
                for batch_idx in range(self.batch_size):
                    batch_states = []
                    batch_nodes = self.trees[agent_idx][batch_idx][stage_num]
                    for node_idx in range(len(batch_nodes)):
                        actions = batch_nodes[node_idx]['actions']
                        batch_states.append(actions)
                    agents[agent_idx].append(batch_states)
            return agents # [Agents][Batch][Node Idx in Stage]
        else:
            raise RuntimeError("Trying to get game state information, but game trees are not defined.")
        
    def get_latest_state(self) -> Dict[Any, List[List[List[Any]]]]: 
        """Get stage state for current stage of the game"""
        return self.get_stage_state(self.stage) # [Agents][Batch][Nodes in Current Stage][World State]
    
    def get_latest_actions(self) -> Dict[Any, List[List[Any]]]: #NOTE(Jari): Here is the method you asked for in case I forget to mention it
        """Get stage state for current stage of the game"""
        return self.get_stage_actions(self.stage) # [Agents][Batch][Node Idx in Current Stage]
    
    def append_actions(self, agent_actions: Dict[Any, List[List[Any]]]) -> List[bool]: 
        """
        Takes outputs generated/made by an agent and appends it to the corresponding nodes
        At a high-level this method performs the following operations:
            1. Updates the generations to appropriate nodes
            2. Create nodes for the next stage and fill their round states as much as possible

        Returns:
            List[bool]: List of bools, where each bool corresponds with a game tree tracked by this agent. 
                        Bool is True if all branches of said game tree are terminal (i.e., this game tree is done) else False 
        """
        if self.trees and len(self.trees) > 0:
            for agent in range(self.swarm_size):
                for batch_idx in range(self.batch_size):
                    for node_idx in range(len(agent_actions[agent][batch_idx])):
                        self.trees[agent][batch_idx].append_node_actions(stage=self.stage, node_idx=node_idx, actions=agent_actions[agent][batch_idx][node_idx]) #Append rollout to current node
                    self.trees[agent][batch_idx].commit_actions_from_stage(stage=self.stage, tree_branching_functions=self.tree_branching_functions) #Signals that the agent is "done" with this tree for this stage+makes children and/or sets nodes as terminal.
        else:
            raise RuntimeError("Trying to update game trees, but game trees are not defined.")

    def advance_stage(self, world_states: Dict[Any, List[List[Tuple[Any]]]]) -> None:
        """
        Updates tree node states of next stage after having communicated, then increments the stage.
        """
        if self.trees and len(self.trees) > 0:
            self.stage += 1 #Increment stage
            for agent in range(self.swarm_size):
                for batch_idx in range(self.batch_size):
                    stage_nodes = self.trees[agent][batch_idx][self.stage]
                    for node_idx in range(len(stage_nodes)):
                        self.trees[agent][batch_idx].append_node_states(stage=self.stage, node_idx=node_idx, states=world_states[agent][batch_idx][node_idx], pruning_functions=self.world_state_pruners) #Assumes new stage's nodes relevant to a parent were created at the moment when we "commited" during append generation
        else:
            raise RuntimeError("Trying to update game trees, but game trees are not defined.")