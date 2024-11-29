import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# Import required libraries
import os
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, set_seed
from datasets import load_dataset
from torch.utils.data import Dataset
from fine_tune import *


class SpeculativeDecoder:
    """
    A class implementing speculative decoding for language models.

    This class uses a larger target model and a smaller draft model to perform
    speculative decoding, potentially speeding up text generation.

    Attributes:
        device (str): The device to run the models on ('cuda' or 'cpu').
        target_model (AutoModelForCausalLM): The larger, more accurate language model.
        draft_model (AutoModelForCausalLM): The smaller, faster language model for draft predictions.
        tokenizer (AutoTokenizer): The tokenizer for both models.
    """


    def __init__(self, target_model_name, draft_model_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the SpeculativeDecoder with target and draft models.

        Args:
            target_model_name (str): The name or path of the target (larger) model.
            draft_model_name (str): The name or path of the draft (smaller) model.
            device (str): The device to run the models on. Defaults to 'cuda' if available, else 'cpu'.
        """

        self.device = device
        self.Mp = AutoModelForCausalLM.from_pretrained(target_model_name).to(self.device)
        self.Mq = AutoModelForCausalLM.from_pretrained(draft_model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        self.no_accepted_tokens = 0
        self.alpha = 0
        self.total_time_taken = 0

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.Mp.eval()
        self.Mq.eval()

    @staticmethod
    def sample(logits, temperature, top_k, top_p):

        """
        Adjust logits for sampling based on temperature, top-k, and top-p parameters.

        Args:
            logits (torch.Tensor): The input logits.
            temperature (float): The temperature for sampling.
            top_k (int): The number of top tokens to consider for top-k sampling.
            top_p (float): The cumulative probability threshold for top-p sampling.

        Returns:
            torch.Tensor: The adjusted probability distribution.
        """

        if temperature <= 1e-6:
            return F.one_hot(logits.argmax(dim=-1), num_classes=logits.size(-1)).float()

        logits = logits / temperature

        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        return F.softmax(logits, dim=-1)

    def generate(self, prompt, temperature=1.0, top_k=0, top_p=1.0, gamma=8, max_new_tokens=50):
        """
        Generate text using speculative decoding.

        Args:
            prompt (str): The input prompt to start generation from.
            temperature (float): The temperature for sampling. Defaults to 1.0.
            top_k (int): The number of top tokens to consider for top-k sampling. Defaults to 0 (disabled).
            top_p (float): The cumulative probability threshold for top-p sampling. Defaults to 1.0 (disabled).
            gamma (int): The number of tokens to generate speculatively in each iteration. Defaults to 5.
            max_new_tokens (int): The maximum number of new tokens to generate. Defaults to 100.

        Returns:
            str: The generated text.
        """
        stime = time.time()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids)

        self.no_accepted_tokens = 0
        betas = []
        all_gammas = []

        for _ in range(0, max_new_tokens, gamma + 1):
            # Generate draft outputs
            all_gammas.append(gamma)
            start_time = time.time()

            with torch.no_grad():
                draft_outputs = self.Mq.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gamma,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            end_time = time.time()

            draft_tokens = draft_outputs.sequences[:, input_ids.size(1):] #torch.Size([1, 5])
            draft_probs = torch.stack(draft_outputs.scores).softmax(-1) #torch.Size([5, 1, 50257]) for GPT2
            #####################
            #Debugging
            # Print draft model tokens and decoded words
#             print(f"Draft Model: {(end_time-start_time):.3f}s", end = " ")
#             for i, token in enumerate(draft_tokens[0]):
#                 word = self.tokenizer.decode(token.item(), skip_special_tokens=True)
#                 print(f"{word}", end = " ")
            #####################

            # Target model single forward pass
            start_time = time.time()
            with torch.no_grad():
                target_outputs = self.Mp(
                    torch.cat([input_ids, draft_tokens], dim=1),
                    attention_mask=torch.cat([attention_mask, torch.ones_like(draft_tokens)], dim=1),
                    return_dict=True,
                )

            end_time = time.time()
#             print(f"\n one run of target model: {end_time-start_time:.3f} s")

            target_logits = target_outputs.logits[:, input_ids.size(1)-1:-1]
            target_probs = self.sample(target_logits, temperature, top_k, top_p)

            # Speculative sampling
            accepted_tokens = []
            for i in range(min(gamma, draft_tokens.size(1))):
                draft_token = draft_tokens[:, i]
                draft_prob = draft_probs[i].gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)
                target_prob = target_probs[:, i].gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)

                accept_prob = torch.min(torch.ones_like(target_prob), target_prob / draft_prob)
                if torch.rand(1, device=self.device) < accept_prob:
                    accepted_tokens.append(draft_token)
                else:
                    break
                # if draft_prob <= target_prob:
                #     # Accept deterministically if draft prob <= target_prob
                #     accepted_tokens.append(draft_token)
                # else:
                #     # Probabilistic rejection if draft prob > target_prob
                #     rejection_prob = 1 - target_prob / draft_prob
                #     if torch.rand(1, device=self.device) < rejection_prob:
                #         break
            # Print accepted tokens
            self.no_accepted_tokens += len(accepted_tokens) + 1
            betas.append(len(accepted_tokens) / gamma)

            #####################
            #Debugging
#             print("\nAccepted:", end = " ")
#             for token in accepted_tokens:
#                 word = self.tokenizer.decode(token.item(), skip_special_tokens=True)
#                 print(f"{word}", end = " ")
            #####################

            num_accepted = len(accepted_tokens)
            acceptance_rate = num_accepted / gamma



            if num_accepted < draft_probs.size(1):
                adjusted_probs = torch.clamp(target_probs[:, num_accepted] - draft_probs[num_accepted], min=0)
                adjusted_probs /= adjusted_probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(adjusted_probs, num_samples=1)
            else:
                next_token = torch.multinomial(target_probs[:, -1], num_samples=1)

            #####################
            #Debugging
#             print(f"\nTarget Model:", end = " ")
#             print(f" {self.tokenizer.decode(next_token.item(), skip_special_tokens=True)}\n")
#             print("-------------------------------")
            ###################

            accepted_tokens.append(next_token)
            new_tokens = torch.cat([token.view(1, 1) for token in accepted_tokens], dim=1)

            input_ids = torch.cat([input_ids, new_tokens], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(new_tokens)], dim=1) #update for next generation

            if input_ids.size(1) - len(self.tokenizer.encode(prompt)) >= max_new_tokens:
                break
#         print(f"\ntotal time taken: {time.time() -stime:.2f} s")
#         print(f"no of tokens produced {self.no_accepted_tokens}")

#         Here run Mp for producing self.no_accepted_tokens and note time
#         total_target_run_time = 0
#         for token in range(self.no_accepted_tokens):
#             start_time = time.time()
#             _ = self.Mp(input_ids)
#             end_time = time.time()
#             total_target_run_time += (end_time - start_time)

        self.alpha = sum(betas) / len(betas)

#         print(f"Total time taken for target model: {total_target_run_time} s")
#         print(f"α = success probability = E(β) = {sum(betas) / len(betas)}")

        # here run Mp for producing no_accepted_tokens and note time
        self.total_time_taken = time.time() - stime;

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True), all_gammas

    def target_generate_greedy(self, prompt, max_new_tokens=50):
        """
        Generate text using standard greedy decoding with the target model.

        Args:
            prompt (str): The input prompt to start generation from.
            max_new_tokens (int): The maximum number of new tokens to generate. Defaults to 50.

        Returns:
            str: The generated text.
        """
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        greedy_output = self.target_model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(greedy_output[0])

    def draft_generate_greedy(self, prompt, max_new_tokens=50):
        """
        Generate text using standard greedy decoding with the draft model.

        Args:
            prompt (str): The input prompt to start generation from.
            max_new_tokens (int): The maximum number of new tokens to generate. Defaults to 50

        Returns:
            str: The generated text.
        """

        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        greedy_output = self.draft_model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(greedy_output[0])
    

model = SpeculativeDecoder(target_model_name='gpt2-large',
                                  draft_model_name='distilgpt2',
                                  device='cuda' if torch.cuda.is_available() else 'cpu')