from unittest import TestCase
import torch

from genrl_swarm.state import GameState


class TestGameState(TestCase):
    def setUp(self) -> None:
        self.batch_size = 5
        round_data = torch.randint(0, 100, (self.batch_size, 10))
        self.state = GameState(0, 0, round_data, None)
        self.swarm_size = len(self.state.outputs)

    def test_append_generation(self) -> None:
        generation = torch.randint(0, 100, (self.swarm_size, self.batch_size, 1, 20)) #generate 20 tokens for each of the 5 batch samples for a single stage
        generation2 = torch.randint(0, 100, (self.swarm_size, self.batch_size, 1, 20))
        self.state.append_generation(generation)
        self.state.append_generation(generation2)
        for i in range(self.swarm_size):
            for j in range(self.batch_size):
                self.assertEqual(generation[i][j][0].tolist(), self.state.outputs[i][j][0][0])
                self.assertEqual(generation2[i][j][0].tolist(), self.state.outputs[i][j][0][1])
                self.assertTrue(len(self.state.outputs[i][j][0]) == 2)
 
        text_generation = [[[[f'this is a string of text {i}']] for i in range(self.batch_size)] for _ in range(self.swarm_size)] #generate 1 string for each of the 5 batch samples for a single stage
        self.state.append_generation(text_generation)
        for i in range(self.swarm_size):
            for j in range(self.batch_size):
                self.assertEqual(text_generation[i][j][0], self.state.outputs[i][j][0][2])
                self.assertTrue(len(self.state.outputs[i][j][0]) == 3)

    def test_advance_stage(self) -> None:
        self.state.advance_stage()
        self.assertEqual(self.state.stage, 1)
        for i in range(self.swarm_size):
            for j in range(self.batch_size):
                self.assertEqual(len(self.state.outputs[i][j]), 2) # each batch has 2 stages now

    def test_advance_round(self) -> None:
        dummy_data = torch.randint(0, 100, (self.batch_size, 10))
        self.state.advance_round(dummy_data)
        self.assertEqual(self.state.round, 1)
        self.assertEqual(self.state.stage, 0)
        self.assertEqual(self.state.batch_size, self.batch_size)
        for i in range(self.swarm_size):
            self.assertEqual(len(self.state.outputs[i]), self.batch_size) # each agent has batch size of 5, so the list should be length 5
            for j in range(self.batch_size):
                self.assertEqual(len(self.state.outputs[i][j]), 1) # each batch's stages have been reset, on 1st stage currently
        
    def test_get_latest(self) -> None:
        # Test with no generations (stage = 0)
        result = self.state.get_latest()
        self.assertIsNone(result)
        
        # Test with tensor generations
        tensor_generation = torch.randint(0, 100, (self.swarm_size, self.batch_size, 1, 20)) #generate 20 tokens for each of the 5 batch samples for a single stage
        self.state.append_generation(tensor_generation)
        self.state.advance_stage()
        
        result = self.state.get_latest()
        self.assertEqual(len(result), self.swarm_size)
        self.assertEqual(len(result[0]), self.batch_size)
        for agent_idx in range(self.swarm_size):
            for batch_idx in range(self.batch_size):
                self.assertEqual(result[agent_idx][batch_idx][0], tensor_generation[agent_idx][batch_idx][0].tolist())
        
        # Test with text/object generations
        text_generations = [[[[(f'This is text {i}', i * 0.1)]] for i in range(self.batch_size)] for _ in range(self.swarm_size)] #generate 1 tuple(text,score) for each of the 5 batch samples for a single stage
        self.state.append_generation(text_generations)
        self.state.advance_stage()
        
        result = self.state.get_latest()
        self.assertEqual(len(result), self.swarm_size)
        for agent_idx in range(self.swarm_size):
            for batch_idx in range(self.batch_size):
                self.assertEqual(result[agent_idx][batch_idx][0], text_generations[agent_idx][batch_idx][0])
        
        single_item_state = GameState(0, 0, ["single_item"], None)
        single_item_state.append_generation([[["generated output"]]])
        single_item_state.advance_stage()
        
        result = single_item_state.get_latest()
        self.assertEqual(result[0][0][0], "generated output")
    
    def test_convert_to_nested_lists(self) -> None:
        #Test on a newly initialized set of outputs
        result = self.state.convert_to_nested_lists()
        expected = [[[[] for stage in self.state.outputs[agent][batch]] for batch in self.batch_size] for agent in range(self.swarm_size)] 
        self.assertEqual(result, expected)

        #Test when there are generations
        generation = torch.randint(0, 100, (self.swarm_size, self.batch_size, 1, 20)) #generate 20 tokens for each of the 5 batch samples for a single stage
        generation2 = torch.randint(0, 100, (self.swarm_size, self.batch_size, 1, 20))
        self.state.append_generation(generation)
        self.state.append_generation(generation2)
        result = self.state.convert_to_nested_lists()
        for i in range(self.swarm_size):
            for j in range(self.batch_size):
                self.assertEqual(self.state.outputs[i][j][0][0], result[i][j][0][0])
                self.assertEqual(self.state.outputs[i][j][0][1], result[i][j][0][1])
                self.assertTrue(len(result[i][j][0]) == 2)
        
        #Test with multiple stages too
        self.state.advance_stage()
        self.state.append_generation(generation)
        self.state.append_generation(generation2)
        result = self.state.convert_to_nested_lists()
        for i in range(self.swarm_size):
            for j in range(self.batch_size):
                self.assertEqual(self.state.outputs[i][j][self.state.stage][0], result[i][j][self.state.stage][0])
                self.assertEqual(self.state.outputs[i][j][self.state.stage][1], result[i][j][self.state.stage][1])
                self.assertTrue(len(result[i][j][self.state.stage]) == 2)

        #test when output in a None
        self.state.outputs = None
        self.assertIsNone(self.state.convert_to_nested_lists())
        
    def test_multi_agent(self) -> None:
        swarm_size = 4
        rank = 2
        round_data = torch.randint(0, 100, (self.batch_size, 10))
        latest_generations = torch.randint(0, 100, (swarm_size, self.batch_size, 1, 10))
        multi_state = GameState(0, 0, round_data, None, swarm_size=swarm_size, rank=rank)
        self.assertEqual(len(multi_state.outputs), swarm_size)
        
        multi_state.append_generation(latest_generations)
        for i in range(swarm_size):
            for j in range(self.batch_size):
                self.assertEqual(latest_generations[i][j][0].tolist(), multi_state.outputs[i][j][0][0])
                self.assertTrue(len(multi_state.outputs[i][j][0]) == 1)
 