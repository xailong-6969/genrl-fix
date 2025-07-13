import contextlib
import gc
import os
from collections import defaultdict
from datetime import datetime
import warnings
from typing import Any, List

import torch
import torch.utils.data
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from trl.data_utils import apply_chat_template
from trl.models import create_reference_model
from trl.trainer.grpo_config import GRPOConfig

from genrl.data import DataManager
from genrl.logging_utils.ml_logger import LoggerMixin
from genrl.rewards import RewardManager
from genrl.state import GameState
from genrl.trainer import TrainerModule


class GRPOLanguageTrainerModule(TrainerModule, LoggerMixin):
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

        self.model = models[
            0
        ]  # TODO(Discuss): How to setup multiple models here? Should be tethered to agent index that'll be given by gamestate. Maybe loop here and add a lil model ID datum to the gamestate?

        # Configuration parameters
        config = kwargs.get("config", None)
        self.args = (
            config
            if isinstance(config, GRPOConfig)
            else GRPOConfig(config) if config else GRPOConfig()
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )

        # Tokenizers
        self.processing_class = kwargs.get("processing_class", None)

        # Additional parameters
        self.callbacks = kwargs.get("callbacks", [])
        self.save_dir = kwargs.get("log_dir", f"./outputs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.global_step = 0
        self.num_generations = kwargs.get("num_generations", 2)
        if self.num_generations <= 1:
            warnings.warn(f"num_generations should be > 1 for GRPO training. Using default value 2.")
            self.num_generations = 2
        self.epsilon = kwargs.get("epsilon", 0.2)
        self.epsilon_high = kwargs.get("epsilon_high", 0.28)
        self.beta = kwargs.get("beta", 0.0)
        self.enable_gradient_checkpointing = kwargs.get(
            "enable_gradient_checkpointing", True
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.autocast = torch.amp.autocast(
                device_type=self.device.type, enabled=self.args.fp16
            )
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.autocast = contextlib.nullcontext()
        else:
            self.device = torch.device("cpu")
            self.autocast = contextlib.nullcontext()

        # Initialize core components
        self._initialize_model(self.enable_gradient_checkpointing)
        self._initialize_tokenizers()
        self._initialize_metrics()
        self._initialize_generation_config()
        self.init_tracker(self.save_dir, log_with=kwargs.get("log_with", None))

    def _initialize_model(self, enable_gradient_checkpointing):
        """Initialize the model and reference model."""
        self.model = self.model.to(self.device)
        if enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Reference model setup
        if self.beta == 0.0:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(self.model).to(self.model.device)

    def _initialize_tokenizers(self):
        """Initialize tokenizers for the model and reward models."""
        if self.processing_class is None:
            self.processing_class = AutoTokenizer.from_pretrained(
                self.model.config._name_or_path, padding_side="left"
            )
            # Ensure pad_token_id is set for generation and padding
            if self.processing_class.pad_token_id is None:
                self.processing_class.pad_token_id = self.processing_class.eos_token_id


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

    def _process_inputs(self, inputs, with_template=True, for_training=False):
        if hasattr(inputs, "to_dict"):
            inputs = [dict(inputs[i]) for i in range(len(inputs))]
        elif isinstance(inputs, dict):
            inputs = [inputs]

        if with_template:
            if for_training:
                templated_prompts = []
                for item in inputs:
                    for _ in range(self.num_generations):
                        templated_prompts.append(
                            apply_chat_template(item, self.processing_class)["prompt"]
                        )
            else:
                templated_prompts = [
                    apply_chat_template(item, self.processing_class)["prompt"]
                    for item in inputs
                ]

        else:
            if for_training:
                templated_prompts = []
                for generations in inputs:
                    # 'generations' here is expected to be a list of strings (completions)
                    for output in generations:
                        templated_prompts.append(output)
            else:
                templated_prompts = [item[0] if isinstance(item, list) else item for item in inputs] # Assuming item is a list of outputs, or just the output string itself. Take first if list.


        input_tokens = self.processing_class(
            text=templated_prompts, return_tensors="pt", padding=True, truncation=True
        )
        return input_tokens

    def generate(
        self, inputs: Any, return_completion_ids: bool = False, stage=0
    ) -> Any:
        """
        Generate outputs from the model for the given inputs.

        Args:
            inputs: Input data for generation
            return_completion_ids: Whether to return completion IDs along with text
            stage: Current stage (0, 1, or 2) for proper output formatting

        Returns:
            Generated outputs in the format expected by the next stage
        """
        input_tokens = self._process_inputs(inputs)
        rollout, rollout_ids = (
            [],
            [],
        )
        for _ in range(self.num_generations):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_tokens.input_ids.to(self.model.device),
                    attention_mask=input_tokens.attention_mask.to(self.model.device),
                    generation_config=self.generation_config,
                )

            # Extract completions (i.e., removes prompt part)
            prompt_length = input_tokens.input_ids.size(1)
            completion_ids = outputs[:, prompt_length:]

            completions = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=True
            )

            if len(rollout) == 0:
                rollout = [[comp] for comp in completions]
                if return_completion_ids:
                    rollout_ids = [[comp] for comp in completion_ids]
            else:
                for idx, comp in enumerate(completions):
                    rollout[idx].append(comp)
                    if return_completion_ids:
                        rollout_ids[idx].append(completion_ids[idx])
        if return_completion_ids:
            return rollout, rollout_ids
        else:
            return rollout

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
        model = model.to(input_ids.device)
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        # The logits_to_keep parameter for model.forward usually means the number of
        # tokens for which to compute the loss (i.e., the completion length).
        # We need logits for sequence_length - 1 tokens to predict tokens from index 1 to sequence_length - 1
        
        # If your model's forward pass doesn't support 'logits_to_keep' directly,
        # you just pass input_ids and attention_mask. The slicing happens after.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        
        logits = logits[
            :, :-1, :
        ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        # If logits_to_keep is provided, slice the logits and labels
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:]
            labels = input_ids[:, 1:][:, -logits_to_keep:] # Labels are shifted by one from input_ids
            loss_mask = attention_mask[:, 1:][:, -logits_to_keep:]
        else: # Calculate over the entire sequence (excluding the first token's prediction)
            labels = input_ids[:, 1:]
            loss_mask = attention_mask[:, 1:]

        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # This part is now handled by the if/else block above.

        # Divide logits by sampling temperature.
        logits = logits / self.args.temperature
        logits_shape = logits.shape
        token_log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits_shape[-1]), # Use reshape for compatibility
            labels.reshape(-1), # Use reshape for compatibility
            reduction="none",
        ).reshape(logits_shape[0], logits_shape[1]) # Reshape back to (B, L_kept)

        # Apply loss mask
        token_log_probs = (
            token_log_probs * loss_mask
            + (1.0 - loss_mask) * torch.finfo(logits.dtype).min
        )
        return token_log_probs  # compute logprobs for the input tokens


    def compute_loss(
        self, model, inputs, num_items_in_batch=1, mode="train", return_metrics=False
    ):
        """Compute the GRPO loss.

        Args:
            model: The model to compute the loss for.
            inputs: The inputs containing prompt_ids, prompt_mask, completion_ids, completion_mask,
                    old_per_token_logps, ref_per_token_logps, and advantages.

        Returns:
            The loss value and metrics.
        """

        # Extract inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )

        # Concatenate prompt and completion
        # These should now be compatible in dim 1 due to the new step function logic
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(self.model.device)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(
            self.model.device
        )
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        # Compute per-token log probabilities
        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )

        # Compute KL divergence between model and reference model if beta > 0
        if self.beta != 0.0:
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, input_ids, attention_mask, logits_to_keep
                )
            else:
                # If ref_model is None but beta > 0, something is wrong with setup
                # For safety, using current model's logps as ref in this specific case
                # This should ideally not happen if beta is set.
                ref_per_token_logps = per_token_logps.clone().detach()

            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        old_per_token_logps = (
            inputs["old_per_token_logps"]
            if (self.args.num_iterations > 1 and inputs["old_per_token_logps"] is not None) # Added check for None
            else per_token_logps.detach()
        )

        # Calculate ratios and loss terms
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(
            coef_1,
            1 - self.epsilon,
            1 + self.epsilon_high if self.epsilon_high is not None else 1 + self.epsilon, # Corrected: 1 + self.epsilon for the upper bound
        )
        advantages = advantages.unsqueeze(dim=-1)

        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Add KL penalty if beta > 0
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Final loss calculation
        # Ensure completion_mask is of the same shape as per_token_loss before multiplication
        # The completion_mask from _process_inputs should already be correctly shaped
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum() if completion_mask.sum() > 0 else torch.tensor(0.0)


        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum() if completion_mask.sum() > 0 else torch.tensor(0.0)
            self._metrics[mode]["kl"].append(mean_kl.item())
        else:
            mean_kl = None # Ensure mean_kl is defined even if beta is 0

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum() if completion_mask.sum() > 0 else torch.tensor(0.0)
        self._metrics[mode]["clip_ratio"].append(clip_ratio.item())
        self._metrics[mode]["loss"].append(loss.item())

        # return for tensorboard
        metrics = {
            "loss": loss.item(),
            "kl": mean_kl.item() if self.beta != 0.0 else 0.0, # Report 0.0 if KL not calculated
            "clip_ratio": clip_ratio.item(),
        }

        if return_metrics:
            return loss, metrics
        else:
            return loss

    def train(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ) -> None:
        """
        Train the model using the given game state and reward manager.

        Args:
            game_state: The current game state.
            reward_manager: The reward manager to use for computing rewards.
        """
        self.model.train()
        global_step = self.global_step
        for stage in range(state.stage):
            global_step = self.step(
                stage, state, data_manager, reward_manager, global_step
            )
        self.global_step = global_step
        self.model.eval()

    def step(
        self,
        stage: int,
        state: GameState,
        data_manager: DataManager,
        reward_manager: RewardManager,
        global_step: int,
    ) -> int:
        global_step += 1

        # Prepare stage's inputs
        stage_inputs_raw = state.get_stage_state(stage)
        stage_inputs_raw, index_mapping = data_manager.prepare_input(stage_inputs_raw, stage)
        assert stage_inputs_raw is not None, f"No inputs found for stage {stage}"
        stage_actions_raw = state.get_stage_actions(stage)
        stage_outputs_raw = [
            stage_actions_raw[index_mapping[idx][0]][index_mapping[idx][1]][index_mapping[idx][2]]
            for idx, _ in enumerate(index_mapping)
        ]
        assert stage_outputs_raw is not None, f"No outputs found for stage {stage}"

        # Combine prompts and completions for consistent tokenization and padding
        combined_texts = []
        original_prompt_lengths = [] # To store the unpadded token length of each prompt

        for i in range(len(stage_inputs_raw)):
            prompt_item = stage_inputs_raw[i]
            # Ensure prompt_str is correctly extracted for templating
            prompt_str = apply_chat_template(prompt_item, self.processing_class)["prompt"]
            
            # Store the original (unpadded) token length of this prompt
            prompt_tokenized_temp = self.processing_class(
                text=[prompt_str], return_tensors="pt", padding=False, truncation=True
            )
            original_prompt_lengths.append(prompt_tokenized_temp.input_ids.size(1))

            # Ensure stage_outputs_raw items are iterable and contain strings
            # If stage_outputs_raw items are lists of strings (e.g., from generate), take each one
            completions_for_item = stage_outputs_raw[i]
            if not isinstance(completions_for_item, list):
                completions_for_item = [completions_for_item] # Wrap in list if not already

            for j in range(self.num_generations):
                if j < len(completions_for_item):
                    completion_str = completions_for_item[j]
                else:
                    # Fallback if not enough completions are provided for num_generations
                    warnings.warn(f"Not enough completions for item {i}. Using empty string for extra generations.")
                    completion_str = "" # Or a special PAD token string if preferred

                combined_texts.append(prompt_str + completion_str)

        # Tokenize all combined prompt+completion sequences at once
        tokenized_combined = self.processing_class(
            text=combined_texts, return_tensors="pt", padding=True, truncation=True
        )

        input_ids_full = tokenized_combined.input_ids.to(self.model.device)
        attention_mask_full = tokenized_combined.attention_mask.to(self.model.device)

        # Now, we need to reconstruct prompt_ids, prompt_mask, completion_ids, completion_mask
        model_inputs = {}
        batch_size_flat, sequence_length_padded = input_ids_full.shape

        prompt_ids_list = []
        prompt_mask_list = []
        completion_ids_list = []
        completion_mask_list = []
        
        # Iterate through each combined sequence in the flattened batch
        for i in range(batch_size_flat):
            # The original prompt length for this specific combined sequence
            # Need to adjust for the flattening: each original input leads to self.num_generations combined texts
            original_prompt_len_for_this_seq = original_prompt_lengths[i // self.num_generations]

            # Extract prompt parts
            # We take tokens up to the original prompt length
            prompt_part_ids = input_ids_full[i, :original_prompt_len_for_this_seq]
            prompt_part_mask = attention_mask_full[i, :original_prompt_len_for_this_seq]
            
            # Extract completion parts
            # Completions start after the original prompt length and go to the end of the padded sequence
            completion_part_ids = input_ids_full[i, original_prompt_len_for_this_seq:]
            completion_part_mask = attention_mask_full[i, original_prompt_len_for_this_seq:]
            
            prompt_ids_list.append(prompt_part_ids)
            prompt_mask_list.append(prompt_part_mask)
            completion_ids_list.append(completion_part_ids)
            completion_mask_list.append(completion_part_mask)

        # Pad prompt and completion tensors independently to the max length within their respective lists
        # This is crucial for creating rectangular tensors for batching
        max_prompt_len_in_batch = max([t.size(0) for t in prompt_ids_list])
        max_completion_len_in_batch = max([t.size(0) for t in completion_ids_list])

        # Stack and pad prompt tensors
        model_inputs["prompt_ids"] = torch.stack([
            torch.cat([p_ids, torch.full((max_prompt_len_in_batch - p_ids.size(0),), self.processing_class.pad_token_id, dtype=p_ids.dtype).to(self.model.device)])
            for p_ids in prompt_ids_list
        ]).to(self.model.device)
        model_inputs["prompt_mask"] = torch.stack([
            torch.cat([p_mask, torch.zeros((max_prompt_len_in_batch - p_mask.size(0),), dtype=p_mask.dtype).to(self.model.device)])
            for p_mask in prompt_mask_list
        ]).to(self.model.device)

        # Stack and pad completion tensors
        model_inputs["completion_ids"] = torch.stack([
            torch.cat([c_ids, torch.full((max_completion_len_in_batch - c_ids.size(0),), self.processing_class.pad_token_id, dtype=c_ids.dtype).to(self.model.device)])
            for c_ids in completion_ids_list
        ]).to(self.model.device)
        model_inputs["completion_mask"] = torch.stack([
            torch.cat([c_mask, torch.zeros((max_completion_len_in_batch - c_mask.size(0),), dtype=c_mask.dtype).to(self.model.device)])
            for c_mask in completion_mask_list
        ]).to(self.model.device)


        # Process rewards with shape fix and participation bonus
        rewards = reward_manager[stage]
        rewards = [
            rewards[index_mapping[idx][0]][index_mapping[idx][1]][index_mapping[idx][2]]
            for idx, _ in enumerate(index_mapping)
        ]
        rewards_2d = []
        for r in rewards:
            if isinstance(r, (int, float)):
                # Add base participation reward (e.g., 1 point per round) and performance
                base_reward = 1.0
                perf_reward = max(0, r)  # Ensure non-negative performance reward
                rewards_2d.append([base_reward + perf_reward] * self.num_generations)
            elif isinstance(r, list) and len(r) == 1:
                rewards_2d.append([1.0 + r[0]] * self.num_generations)
            elif isinstance(r, list) and len(r) > 1:
                rewards_2d.append([1.0 + val for val in r[:self.num_generations]])
            else:
                # Handle cases where reward might be None or unexpected format
                warnings.warn(f"Unexpected reward format encountered: {r}. Defaulting to [1.0] * num_generations.")
                rewards_2d.append([1.0] * self.num_generations)

        # Ensure rewards_2d is not empty before converting to tensor
        if not rewards_2d:
            warnings.warn("No rewards processed. Skipping loss calculation for this step.")
            return global_step # Or handle as an error if this state is invalid

        rewards = torch.tensor(rewards_2d, dtype=torch.float32) # Specify dtype for consistency

        assert rewards is not None, f"No rewards found for stage {stage}"
        assert rewards.dim() == 2 and rewards.size(1) == self.num_generations, \
            f"Rewards shape {rewards.shape} does not match expected [batch_size, {self.num_generations}]"

        with torch.no_grad():
            # Handle potential division by zero if std is 0
            std_rewards = rewards.std(dim=1, keepdim=True)
            advantages = rewards - rewards.mean(dim=1, keepdim=True)
            if rewards.shape[1] > 1 and torch.any(std_rewards > 1e-8): # Only normalize if std > 0
                advantages /= std_rewards + 1e-8
        
        advantages = torch.flatten(advantages).to(self.model.device)
        model_inputs["advantages"] = advantages.squeeze(dim=-1) # Ensure it's 1D
        model_inputs["old_per_token_logps"] = None

        with self.autocast:
            loss = self.compute_loss(self.model, model_inputs)

        # Only perform backward pass and optimizer step if loss is not NaN/Inf and is not 0
        if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() != 0.0:
            loss.backward()
            self.optimizer.step()
            self.model.zero_grad()
        else:
            warnings.warn(f"Skipping backward pass due to invalid loss: {loss.item()}")


        metrics = {"train/loss": loss.cpu().mean().item() if not torch.isnan(loss) and not torch.isinf(loss) else 0.0}
        metrics.update({"train/rewards": rewards.cpu().mean().item()})
        self.log(metrics, global_step)

        self.cleanup_step()

        return global_step

    @torch.no_grad()
    def evaluate(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ):
        pass

    def save(self, save_dir: str) -> None:
        """
        Save the model and trainer state to the given directory.

        Args:
            save_dir: The directory to save to.
        """
        os.makedirs(save_dir, exist_ok=True)
        # Assuming self.model is a HuggingFace model or has a save_pretrained method
        self.model.save_pretrained(save_dir)
        self.processing_class.save_pretrained(save_dir) # Save tokenizer

        # Save additional state
        torch.save(
            {
                "metrics": self._metrics,
                "total_train_tokens": self._total_train_tokens,
                "generation_config": self.generation_config.to_dict(), # Save config as dict
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
            },
            os.path.join(save_dir, "trainer_state.pt"),
        )

    @classmethod
    def load(cls, load_dir: str) -> "GRPOLanguageTrainerModule":
        """
        Load a trainer module from the given directory.

        Args:
            load_dir: The directory to load from.

        Returns:
            The loaded trainer module.
        """
        # Load model
        model = AutoModelForCausalLM.from_pretrained(load_dir)
        tokenizer = AutoTokenizer.from_pretrained(load_dir)

        # Create trainer instance
        # Pass the loaded tokenizer as processing_class
        trainer = cls([model], processing_class=tokenizer)

        # Load additional state
        trainer_state_path = os.path.join(load_dir, "trainer_state.pt")
        if os.path.exists(trainer_state_path):
            trainer_state = torch.load(trainer_state_path)
            trainer._metrics = trainer_state.get("metrics", defaultdict(list))
            trainer._total_train_tokens = trainer_state.get("total_train_tokens", 0)
            
            # Load generation config
            gen_config_dict = trainer_state.get("generation_config")
            if gen_config_dict:
                trainer.generation_config = GenerationConfig.from_dict(gen_config_dict)
            else:
                trainer._initialize_generation_config() # Re-initialize if not found

            # Load optimizer state
            optimizer_state_dict = trainer_state.get("optimizer_state_dict")
            if optimizer_state_dict:
                trainer.optimizer.load_state_dict(optimizer_state_dict)
            
            trainer.global_step = trainer_state.get("global_step", 0)

        return trainer

    def cleanup_step(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    def cleanup(self):
        self.cleanup_trackers()

