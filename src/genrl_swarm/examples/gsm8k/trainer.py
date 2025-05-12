
import os
from collections import defaultdict
from typing import Any, List

import torch
import torch.utils.data
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    Trainer,
)

from genrl_swarm.trainer import TrainerModule
from genrl_swarm.rewards import RewardManager
from genrl_swarm.data import DataManager
from genrl_swarm.state import GameState
from genrl_swarm.logging_utils.tensorboard_logger import TensorboardLoggerMixin
from trl.trainer.utils import selective_log_softmax
from trl.trainer.grpo_config import GRPOConfig
from trl.models import create_reference_model


class GRPOTrainerModule(TrainerModule, TensorboardLoggerMixin):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method.
    Implements the TrainerModule interface defined in base_trainer.py.
    """
    
    def __init__(self, models: List[Any], **kwargs):
        """
        Initialize the GRPO trainer module.
        
        Args:
            models: List containing the model to be trained.
            **kwargs: Additional arguments for configuration.
        """
        # Extract model and reward functions
        if not models or len(models) < 1:
            raise ValueError("At least one model must be provided")
        
        self.model = models[0] #TODO(johnny): how to pick model?
        
        # Configuration parameters
        config = kwargs.get("config", None)
        self.args = config if isinstance(config, GRPOConfig) else GRPOConfig(config) if config else GRPOConfig()
        
        # Tokenizers
        self.processing_class = kwargs.get("processing_class", None)
        
        # Additional parameters
        self.callbacks = kwargs.get("callbacks", [])
        self.save_dir = kwargs.get("save_dir", "./output")
        
        # Initialize core components
        self._initialize_model()
        self._initialize_tokenizers()
        self._initialize_metrics()
        self._initialize_generation_config()
        self._initialize_trainer()
        self.init_tracker(self.save_dir)

    
    def _initialize_model(self):
        """Initialize the model and reference model."""
        # Set up model hyperparameters
        self.beta = self.args.beta
        
        # Reference model setup
        if self.beta == 0.0:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(self.model)
    
    def _initialize_tokenizers(self):
        """Initialize tokenizers for the model and reward models."""
        if self.processing_class is None:
            self.processing_class = AutoTokenizer.from_pretrained(
                self.model.config._name_or_path, 
                padding_side="left"
            )
    
    def _initialize_metrics(self):
        """Initialize metrics tracking for training and evaluation."""
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
    
    def _initialize_generation_config(self):
        # Set generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=self.args.max_completion_length,
            do_sample=True,
            pad_token_id=self.processing_class.pad_token_id,
            bos_token_id=self.processing_class.bos_token_id,
            eos_token_id=self.processing_class.eos_token_id,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            min_p=self.args.min_p,
            repetition_penalty=self.args.repetition_penalty,
        )

    def _initialize_trainer(self):
        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            tokenizer=self.processing_class,
            callbacks=self.callbacks,
        )
        self.trainer.compute_loss = self.compute_loss
    
    def generate(self, inputs: Any) -> Any:
        if isinstance(inputs, str):
            inputs = [inputs]
        
        input_tokens = self.processing_class(
            text=inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_prompt_length,
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_tokens.input_ids,
                attention_mask=input_tokens.attention_mask,
                generation_config=self.generation_config,
            )
            
        # Extract completions (remove prompt part)
        prompt_length = input_tokens.input_ids.size(1)
        completion_ids = outputs[:, prompt_length:]
        
        completions = self.processing_class.batch_decode(
            completion_ids, 
            skip_special_tokens=True
        )
        
        return completions, completion_ids
    
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        """Get the per-token log probabilities for the input tokens.
        
        Args:
            model: The model to compute log probabilities for.
            input_ids: The input token IDs.
            attention_mask: The attention mask.
            logits_to_keep: The number of logits to keep.
            
        Returns:
            The per-token log probabilities.
        """
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        logits = logits[:, -logits_to_keep:]
        # Divide logits by sampling temperature.
        logits = logits / self.args.temperature
        return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, mode='train'):
        """Compute the GRPO loss.
        
        Args:
            model: The model to compute the loss for.
            inputs: The inputs containing prompt_ids, prompt_mask, completion_ids, completion_mask,
                    old_per_token_logps, ref_per_token_logps, and advantages.
            return_outputs: Whether to return the outputs in addition to the loss.
            
        Returns:
            The loss value, and optionally the outputs.
        """
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
        # Extract inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        
        # Concatenate prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        
        # Compute per-token log probabilities
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        
        # Compute KL divergence between model and reference model if beta > 0
        if self.beta != 0.0:
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, input_ids, attention_mask, logits_to_keep)
            else:
                ref_per_token_logps = per_token_logps.clone()

            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
        
        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        old_per_token_logps = inputs["old_per_token_logps"] if self.args.num_iterations > 1 else per_token_logps.detach()
        
        # Calculate ratios and loss terms
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.args.epsilon, 1 + self.args.epsilon_high if self.args.epsilon_high is not None else self.args.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        # Add KL penalty if beta > 0
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        
        # Final loss calculation
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        
        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(mean_kl.item())
        
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(clip_ratio.item())
        self._metrics[mode]["loss"].append(loss.item())

        # return for tensorboard
        metrics = {
            "loss": loss.item(),
            "kl": mean_kl.item() if self.beta != 0.0 else None,
            "clip_ratio": clip_ratio.item(),
            }
        return loss, metrics

    def train(self, game_state: GameState, reward_manager: RewardManager) -> None:
        """
        Train the model using the given game state and reward manager.
        
        Args:
            game_state: The current game state.
            reward_manager: The reward manager to use for computing rewards.
        """
        self.model.train()
        global_step = self.global_step
        global_step = self.step(game_state, reward_manager, global_step)
        self.global_step = global_step
        self.model.eval()

    def step(self, game_state: GameState, reward_manager: RewardManager, global_step: int) -> int:
        global_step += 1        

        model_inputs = game_state.get_latest()

        rewards = reward_manager(game_state.round, game_state.stage, game_state)

        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8) #TODO(johnny): handle multi-turn
        model_inputs["advantages"] = advantages
        model_inputs["old_per_token_logps"] = None

        do_sync_step = global_step % self.args.gradient_accumulation_steps == 0
        self.trainer.accelerator.gradient_state._set_sync_gradients(do_sync_step)
        
        loss, metrics = self.trainer.training_step(self.model, model_inputs)
        
        if do_sync_step:
            # Since we perform prefetching, we need to manually set sync_gradients to True
            self.trainer.accelerator.gradient_state._set_sync_gradients(True)

            # Gradient clipping
            if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                if is_sagemaker_mp_enabled() and self.args.fp16:
                    _grad_norm = self.trainer.optimizer.clip_master_grads(self.args.max_grad_norm)
                elif self.trainer.use_apex:
                    # Revert to normal clipping otherwise, handling Apex or full precision
                    _grad_norm = nn.utils.clip_grad_norm_(
                        amp.master_params(self.trainer.optimizer),
                        self.args.max_grad_norm,
                    )
                else:
                    _grad_norm = self.trainer.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm,
                    )

                if (
                    is_accelerate_available()
                    and self.trainer.accelerator.distributed_type == DistributedType.DEEPSPEED
                ):
                    grad_norm = self.model.get_global_grad_norm()
                    # In some cases the grad norm may not return a float
                    if hasattr(grad_norm, "item"):
                        grad_norm = grad_norm.item()
                else:
                    grad_norm = _grad_norm

            self.optimizer.step()

            if not self.trainer.accelerator.optimizer_step_was_skipped:
                # Delay optimizer scheduling until metrics are generated
                if not isinstance(self.trainer.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.trainer.lr_scheduler.step()

            self.model.zero_grad()
        
        self.log(metrics, global_step)

        return global_step

    @torch.no_grad()
    def evaluate(self, data_manager: DataManager, reward_manager: RewardManager):
        self.model.eval()
        eval_data_loader = data_manager.eval_data()

        batch = next(iter(eval_data_loader))

        completions, completion_ids = self.model.generate(batch['prompts'])

        reward_fn = reward_manager.dispatch_reward_fn(reward_manager.round, reward_manager.stage)
        rewards = reward_fn.evaluation(batch['prompts'], completions)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        batch['completion_ids'] = completion_ids
        batch['completion_mask'] = completion_ids != self.tokenizer.eos_token_id
        batch['advantages'] = advantages
        batch['old_per_token_logps'] = None

        loss, metrics = self.compute_loss(batch, mode='eval')
        self.log(metrics, reward_manager.global_step)
    
    def save(self, save_dir: str) -> None:
        """
        Save the model and trainer state to the given directory.
        
        Args:
            save_dir: The directory to save to.
        """
        os.makedirs(save_dir, exist_ok=True)
        self.trainer.save_model(save_dir)
        
        # Save additional state
        torch.save({
            "metrics": self._metrics,
            "total_train_tokens": self._total_train_tokens,
            "generation_config": self.generation_config,
        }, os.path.join(save_dir, "trainer_state.pt"))
    
    @classmethod
    def load(cls, load_dir: str) -> 'GRPOTrainerModule':
        """
        Load a trainer module from the given directory.
        
        Args:
            load_dir: The directory to load from.
            
        Returns:
            The loaded trainer module.
        """
        # Load model
        model = AutoModelForCausalLM.from_pretrained(load_dir)
        
        # Create trainer instance
        trainer = cls([model])
        
        # Load additional state
        trainer_state = torch.load(os.path.join(load_dir, "trainer_state.pt"))
        trainer._metrics = trainer_state["metrics"]
        trainer._total_train_tokens = trainer_state["total_train_tokens"]
        trainer.generation_config = trainer_state["generation_config"]
        
        return trainer
