import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from fine_tune import *

class SpeculativeDecoderTriple:
    def __init__(self, target_model_name, draft_model_name, subdraft_model_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.target_model = AutoModelForCausalLM.from_pretrained(target_model_name).to(self.device)
        self.draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name).to(self.device)
        self.subdraft_model = AutoModelForCausalLM.from_pretrained(subdraft_model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.no_accepted_tokens = 0
        self.alpha = 0 

        self.target_model.eval()
        self.draft_model.eval()
        self.subdraft_model.eval()

    @staticmethod
    def sample(logits, temperature=1.0, top_k=0, top_p=1.0):
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

    def generate(self, prompt, temperature=1.0, top_k=0, top_p=1.0, gamma=4, max_new_tokens=100):
        stime = time.time()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids)
        betas_draft = []
        betas_target = []

        for _ in range(0, max_new_tokens, gamma + 1):
            with torch.no_grad():
                # Step 1: Generate subdraft tokens
                ####################
                subdraft_outputs = self.subdraft_model.generate(
                    input_ids, attention_mask=attention_mask, max_new_tokens=gamma,
                    do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p,
                    return_dict_in_generate=True, output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                subdraft_tokens = subdraft_outputs.sequences[:, input_ids.size(1):]
                subdraft_probs = torch.stack(subdraft_outputs.scores).softmax(-1)
#                 print(subdraft_tokens.shape)
#                 print(subdraft_probs.shape)
#                 print("---------------------------")
#                 print("Subdraft model:", end = " ")
#                 for i, token in enumerate(subdraft_tokens[0]):
#                     word = self.tokenizer.decode(token.item(), skip_special_tokens=True)
#                     print(f"{word}", end=" ")
#                 print()
#                 print("---------------------------")
#                 print()
                
                
                
                # Step 2: Validate subdraft tokens with draft model
                draft_outputs = self.draft_model(
                    torch.cat([input_ids, subdraft_tokens], dim=1),
                    attention_mask=torch.cat([attention_mask, torch.ones_like(subdraft_tokens)], dim=1),
                    return_dict=True,
                )
                draft_logits = draft_outputs.logits[:, input_ids.size(1)-1:-1]
                draft_probs = self.sample(draft_logits, temperature, top_k, top_p)
                
#                 print("draft_logits", draft_logits.shape)
#                 print("draft_probs", draft_probs.shape)
                ###############
                #accept reject from draft model
                ##############
                
                accepted_tokens_draft = []
                for i in range(min(gamma, subdraft_tokens.size(1))):
                    subdraft_token = subdraft_tokens[:, i]
                    subdraft_prob = subdraft_probs[i].gather(-1, subdraft_token.unsqueeze(-1)).squeeze(-1)
                    draft_prob = draft_probs[:, i].gather(-1, subdraft_token.unsqueeze(-1)).squeeze(-1)
                    if subdraft_prob == 0 or draft_prob == 0:
#                         print(f"Skipping token at index {i} due to zero probability.")
                        continue 
                    accept_prob = torch.min(torch.ones_like(draft_prob), draft_prob / subdraft_prob)
#                     print(f"accept_prob at index {i}: {accept_prob}")
                    if torch.rand(1, device=self.device) < accept_prob:
                        accepted_tokens_draft.append(subdraft_token)
#                         print(f"Accepted subdraft token at index {i}: {subdraft_token}")
                    else:
#                         print(f"Rejected subdraft token at index {i}.")
                        break
#                 print(f"Number of accepted tokens in draft: {len(accepted_tokens_draft)}")
                 
                #####################
                #Debugging
#                 print("\nAccepted by draft:", end = " ")
#                 for token in accepted_tokens_draft:
#                     word = self.tokenizer.decode(token.item(), skip_special_tokens=True)
#                     print(f"{word}", end = " ")
#                 print()
                #####################
                
                betas_draft.append(len(accepted_tokens_draft) / gamma)
                num_accepted_draft = len(accepted_tokens_draft)
#                 print("draft beta: ",betas_draft[-1])
                
                if num_accepted_draft < subdraft_probs.size(1):
                    adjusted_probs = torch.clamp(draft_probs[:, num_accepted_draft] - subdraft_probs[num_accepted_draft], min=0)
                    adjusted_probs /= adjusted_probs.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(adjusted_probs, num_samples=1)
                else:
                    next_token = torch.multinomial(draft_probs[:, -1], num_samples=1)
                    
                #####################
                #Debugging
#                 print(next_token.shape)
#                 print(f"\Draft Model:", end = " ")
#                 print(f" {self.tokenizer.decode(next_token.item(), skip_special_tokens=True)}\n")
#                 print("-------------------------------")
                ###################
                
                accepted_tokens_draft.append(next_token)
                new_tokens_draft = torch.cat([token.view(1, 1) for token in accepted_tokens_draft], dim=1)

                input_ids_target = torch.cat([input_ids, new_tokens_draft], dim=1)
#                 attention_mask = torch.cat([attention_mask, torch.ones_like(new_tokens_draft)], dim=1) #update for next generation
                
#                 accepted_tokens_draft = torch.cat(accepted_tokens_draft, dim=1) if accepted_tokens_draft else None
                
#                 print("after adding new token: ", accepted_tokens_draft.shape)
                new_gamma = len(accepted_tokens_draft)
            
                
#                 accepted_tokens_draft_tensor = torch.stack(accepted_tokens_draft).unsqueeze(0)

#                 print(accepted_tokens_draft_tensor.shape)
                ################
                # Done...........
                ################
                
                # Step 3: Validate accepted draft tokens with target model
                # Target Model
                ##################
                
                
                # Combine accepted tokens into a tensor
#                 accepted_tokens_draft_tensor = torch.cat(accepted_tokens_draft, dim=1)
                target_inputs = torch.cat([input_ids_target, new_tokens_draft], dim=1)
                target_attention_mask = torch.cat([attention_mask, torch.ones_like(new_tokens_draft)], dim=1)

                # Pass to target model
                target_outputs = self.target_model(
                    target_inputs,
                    attention_mask=target_attention_mask,
                    return_dict=True,
                )
                target_logits = target_outputs.logits[:, input_ids.size(1)-1:-1]
                target_probs = self.sample(target_logits, temperature, top_k, top_p)
                
                

                # Validate tokens with target model
                
                accepted_tokens_target = []
                for i in range(new_tokens_draft.size(1)):
                    draft_token = new_tokens_draft[:, i]

                    # Ensure the dimensions of draft_probs and target_probs are correct
                    if draft_probs.size(1) <= i or target_probs.size(1) <= i:
#                         print(f"Skipping index {i}: draft_probs or target_probs out of bounds.")
                        break

                    # Compute probabilities
                    draft_prob = draft_probs[:, i].gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)
                    target_prob = target_probs[:, i].gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)
                    
                    # Calculate acceptance probability
                    accept_prob = torch.min(torch.ones_like(target_prob), target_prob / draft_prob)

                    # Accept or reject the token
                    if torch.rand(1, device=self.device) < accept_prob:
                        accepted_tokens_target.append(draft_token)
                    else:
                        break

                # Debugging output
#                 print(f"Accepted tokens in target model: {len(accepted_tokens_target)}")

#                 print("target accepted: ", len(accepted_tokens_target))
                betas_target.append(len(accepted_tokens_target) / new_gamma)
#                 print("target beta: ", betas_target[-1])

                num_accepted_by_target = len(accepted_tokens_target)


                if num_accepted_by_target < target_probs.size(1) and num_accepted_by_target < draft_probs.size(1):
                    adjusted_probs = torch.clamp(
                        target_probs[:, num_accepted_by_target] - draft_probs[:, num_accepted_by_target],
                        min=0
                    )
                    adjusted_probs /= adjusted_probs.sum(dim=-1, keepdim=True)
                    next_token_target = torch.multinomial(adjusted_probs, num_samples=1)
                else:
                    next_token_target = torch.multinomial(target_probs[:, -1], num_samples=1)


                
                accepted_tokens_target.append(next_token_target)
                new_tokens_target = torch.cat([token.view(1, 1) for token in accepted_tokens_target], dim=1)

                input_ids = torch.cat([input_ids, new_tokens_target], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(new_tokens_target)], dim=1) #update for next generation
                #####################
                #Debugging
#                 print(f"\nTarget Model:", end = " ")
#                 print(f" {self.tokenizer.decode(next_token_target.item(), skip_special_tokens=True)}\n")
#                 print("-------------------------------")
                ###################
                
#                 if accepted_tokens_target:
#                     accepted_tokens_target_tensor = torch.cat(accepted_tokens_target, dim=1)
#                     input_ids = torch.cat([input_ids, accepted_tokens_target_tensor], dim=1)
#                     attention_mask = torch.cat([attention_mask, torch.ones_like(accepted_tokens_target_tensor)], dim=1)

            if input_ids.size(1) - len(self.tokenizer.encode(prompt)) >= max_new_tokens:
                break

        self.alpha_draft = sum(betas_draft) / len(betas_draft) if betas_draft else 0
        self.alpha_target = sum(betas_target) / len(betas_target) if betas_target else 0
#         print(f"Total time taken: {time.time() - stime:.2f} s")
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
# Initialize the multi-level speculative decoder
spec_decoder = SpeculativeDecoderTriple(
    target_model_name="gpt2-large",
    draft_model_name="gpt2",
    subdraft_model_name="distilgpt2"
)

seed = 42
max_length = 128
set_seed(seed)
dataset = load_dataset('cnn_dailymail', '3.0.0')

val_data = dataset['validation']
test_data = dataset['test']
val_data = val_data.select(range(3000))
test_data = test_data.select(range(50))

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

val_dataset = CNNDailyMailGPT2Dataset(val_data, max_length)
test_dataset = CNNDailyMailGPT2Dataset(test_data, max_length)


import time
import numpy as np
from tqdm import tqdm

def evaluate_speculative_decoding(dataset, spec_decoder, max_new_tokens=100, temperature=1.0, top_k=0, top_p=1.0, gamma=5, output_file="evaluation_results.txt"):
    total_time = 0
    total_target_model_time = 0
    total_tokens_produced = 0
    alphas_draft = []
    alphas_target = []

    # Open file to write outputs
    with open(output_file, "w") as f:
        # Loop through each sample in the dataset
        temp = 0
        for i, sample in enumerate(tqdm(dataset, desc="Evaluating Speculative Decoding")):
            prompt = sample['input_text']  # Modify prompt as needed

            # Perform speculative decoding and measure time and alpha
            start_time = time.time()
            generated_text = spec_decoder.generate(prompt, temperature=temperature, top_k=top_k, top_p=top_p, gamma=gamma, max_new_tokens=max_new_tokens)
            # f.write(f"Generated Text for Sample {i}:\n{generated_text}\n\n")
            temp += len(generated_text.split())
            decoding_time = time.time() - start_time

            # Collect timing metrics and alpha (success probability)
            total_time += decoding_time
            total_tokens_produced += spec_decoder.no_accepted_tokens  # Assuming no_accepted_tokens is updated in spec_decoder
            alphas_draft.append(spec_decoder.alpha_draft)
            alphas_target.append(spec_decoder.alpha_target)

        # Calculate overall metrics
        avg_time_per_sample = total_time / len(dataset)
        avg_tokens_per_sample = total_tokens_produced / len(dataset)
        avg_alpha_draft = np.mean(alphas_draft)
        avg_alpha_target = np.mean(alphas_target)
        # Write summary metrics to file
        print("\nEvaluation Results on Test Dataset:\n")
#         print(f"Average time per sample: {avg_time_per_sample:.2f} seconds\n")
#         print(f"Average tokens per sample: {avg_tokens_per_sample:.2f}\n")
        print(f"average token by speculative {total_time / temp}")
        print(f"Average alpha (acceptance probability) Draft: {avg_alpha_draft:.2f}\n")
        print(f"Average alpha (acceptance probability) Target: {avg_alpha_target:.2f}\n")

    return {
        "avg_time_per_sample": avg_time_per_sample,
        "avg_tokens_per_sample": avg_tokens_per_sample,
    }

evaluate_speculative_decoding(test_dataset, spec_decoder)