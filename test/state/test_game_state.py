from unittest import TestCase
import torch

from genrl_swarm.state import GameState


class TestGameState(TestCase):
    def setUp(self) -> None:
        self.batch_size = 5
        train_batch = torch.randint(0, 100, (self.batch_size, 10))
        self.state = GameState(0, 0, train_batch, None)
        self.swarm_size = len(self.state.outputs)

    def test_append_generation(self) -> None:
        generation = torch.randint(0, 100, (self.batch_size, 20)) #generate 20 tokens for each of the 5 batch samples
        generation2 = torch.randint(0, 100, (self.batch_size, 20))
        self.state.append_generation(generation)
        self.state.append_generation(generation2)
        for i in range(self.swarm_size):
            for j in range(self.batch_size):
                self.assertEqual(generation[j].tolist(), self.state.outputs[i][j][0][0])
                self.assertEqual(generation2[j].tolist(), self.state.outputs[i][j][0][1])
                self.assertTrue(len(self.state.outputs[i][j][0]) == 2)
 
        text_generation = [[f'this is a string of text {i}'] for i in range(self.batch_size)]
        self.state.append_generation(text_generation)
        for i in range(self.swarm_size):
            for j in range(self.batch_size):
                self.assertEqual(text_generation[j], self.state.outputs[i][j][0][2])
                self.assertTrue(len(self.state.outputs[i][j][0]) == 3)

    def test_advance_stage(self) -> None:
        self.state.advance_stage()
        self.assertEqual(self.state.stage, 1)
        for i in range(self.swarm_size):
            for j in range(self.batch_size):
                self.assertEqual(len(self.state.outputs[i][j]), 2) # each batch has 2 stages now

    def test_advance_round(self) -> None:
        self.state.advance_round()
        self.assertEqual(self.state.round, 1)
        self.assertEqual(self.state.stage, 0)
        for i in range(self.swarm_size):
            for j in range(self.batch_size):
                self.assertEqual(len(self.state.outputs[i][j]), 1) # each batch's stages have been reset, on 1st stage currently
        