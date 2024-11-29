import matplotlib.pyplot as plt
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

    def generate(self, prompt, temperature=1.0, top_k=0, top_p=1.0, initial_gamma=3, max_new_tokens=50):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids)

        gamma = initial_gamma  # Start with initial gamma value
        min_gamma, max_gamma = 1, 10  # Define min and max bounds for gamma
        acceptance_threshold_high, acceptance_threshold_low = 0.8, 0.3  # Thresholds for adjusting gamma
        betas = []
        gamma_values = []  # List to store gamma values at each step
        self.no_accepted_tokens = 0

        for _ in range(0, max_new_tokens, gamma + 1):
            gamma_values.append(gamma)  # Append current gamma value

            # Draft model generation
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

            draft_tokens = draft_outputs.sequences[:, input_ids.size(1):]
            draft_probs = torch.stack(draft_outputs.scores).softmax(-1)

            # Target model forward pass
            with torch.no_grad():
                target_outputs = self.Mp(
                    torch.cat([input_ids, draft_tokens], dim=1),
                    attention_mask=torch.cat([attention_mask, torch.ones_like(draft_tokens)], dim=1),
                    return_dict=True,
                )

            target_logits = target_outputs.logits[:, input_ids.size(1)-1:-1]
            target_probs = self.sample(target_logits, temperature, top_k, top_p)

            # Speculative sampling and acceptance calculation
            accepted_tokens = []
            num_accepted = 0
            for i in range(min(gamma, draft_tokens.size(1))):
                draft_token = draft_tokens[:, i]
                draft_prob = draft_probs[i].gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)
                target_prob = target_probs[:, i].gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)

                accept_prob = torch.min(torch.ones_like(target_prob), target_prob / draft_prob)
                if torch.rand(1, device=self.device) < accept_prob:
                    accepted_tokens.append(draft_token)
                    num_accepted += 1
                else:
                    break

            # Adjust gamma based on acceptance rate
            self.no_accepted_tokens += len(accepted_tokens) + 1
            acceptance_rate = num_accepted / gamma
            betas.append(acceptance_rate)
            
            if acceptance_rate > acceptance_threshold_high and gamma < max_gamma:
                gamma += 1
            elif acceptance_rate < acceptance_threshold_low and gamma > min_gamma:
                gamma -= 1

            # Append accepted tokens and move to the next input sequence
            if num_accepted < draft_probs.size(1):
                accepted_tokens.append(draft_tokens[:, num_accepted])
            else:
                next_token = torch.multinomial(target_probs[:, -1], num_samples=1)
                accepted_tokens.append(next_token)

            new_tokens = torch.cat([token.view(1, 1) for token in accepted_tokens], dim=1)

            input_ids = torch.cat([input_ids, new_tokens], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(new_tokens)], dim=1)

            if input_ids.size(1) - len(self.tokenizer.encode(prompt)) >= max_new_tokens:
                break

        self.alpha = sum(betas) / len(betas)
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True), gamma_values


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



import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d  # For smoothing

def evaluate_speculative_decoding(dataset, spec_decoder, max_new_tokens=100, temperature=1.0, top_k=0, top_p=1.0, gamma=3):
    total_time = 0
    total_tokens_produced = 0
    alphas = []
    all_gamma_values = []  # Store gamma values across all samples

    # Loop through each sample in the dataset
    for i, sample in enumerate(tqdm(dataset, desc="Evaluating Speculative Decoding")):
        prompt = sample['input_text']  # Modify prompt as needed

        # Perform speculative decoding and measure time and alpha
        start_time = time.time()
        generated_text, gamma_values = spec_decoder.generate(prompt, temperature=temperature, top_k=top_k, top_p=top_p, initial_gamma=gamma, max_new_tokens=max_new_tokens)
        decoding_time = time.time() - start_time

        total_time += decoding_time
        total_time = total_time / spec_decoder.no_accepted_tokens
#         total_tokens_produced += spec_decoder.no_accepted_tokens
        alphas.append(spec_decoder.alpha)
        all_gamma_values.extend(gamma_values)  # Collect gamma values for plotting

    # Calculate overall metrics
#     avg_time_per_sample = total_time / len(dataset)
#     avg_tokens_per_sample = total_tokens_produced / len(dataset)
    avg_alpha = np.mean(alphas)
    
    temp = total_time / len(dataset)
    print("avg time taken by speculative decoding: ", temp)
#     print(f"\nEvaluation Results on Test Dataset:")
#     print(f"Average time per sample: {avg_time_per_sample:.2f} seconds")
    print(f"Average alpha (acceptance probability): {avg_alpha:.2f}")
#     print(f"Average time per token: {total_time / total_tokens_produced:.4f} seconds")
    

    # Smoothing gamma values for a cleaner plot
    smoothed_gamma_values = gaussian_filter1d(all_gamma_values, sigma=2)  # Adjust sigma for more/less smoothing

    # Plot gamma values
    plt.figure(figsize=(12, 6))
    plt.plot(smoothed_gamma_values, label="Smoothed Gamma Value", color='blue', linewidth=2)
    plt.scatter(range(len(all_gamma_values)), all_gamma_values, s=10, color='red', alpha=0.6, label="Original Gamma Values")
    plt.axhline(np.mean(all_gamma_values), color='green', linestyle='--', linewidth=1, label="Mean Gamma Value")
    plt.grid(alpha=0.3)
    plt.xlabel("Generation Step", fontsize=12)
    plt.ylabel("Gamma", fontsize=12)
    plt.title("Change in Gamma Value Over Generation Steps", fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    return {
        "avg_time_per_sample": avg_time_per_sample,
        "avg_tokens_per_sample": avg_tokens_per_sample,
        "avg_alpha": avg_alpha,
    }


# Run evaluation
results = evaluate_speculative_decoding(test_dataset, model)