import torch
import torch.nn as nn

class UniversalPromptInjectionLLM:
    def __init__(self, model, tokenizer, epsilon=0.1, steps=10, lr=0.01, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.epsilon = epsilon
        self.steps = steps
        self.lr = lr
        self.device = device

    def generate_injection(self, target_text):

        target_ids = self.tokenizer.encode(target_text, return_tensors="pt").to(self.device)

        injection = torch.randn((1, target_ids.size(1), self.model.config.hidden_size),requires_grad=True, device=self.device)

        optimizer = torch.optim.Adam([injection], lr=self.lr)

        loss_fn = nn.CrossEntropyLoss()

        for step in range(self.steps):
            optimizer.zero_grad()

            outputs = self.model(inputs_embeds=injection)
            logits = outputs.logits.squeeze(0)

            loss = loss_fn(logits, target_ids.squeeze(0))

            loss.backward()

            optimizer.step()

            with torch.no_grad():
                injection.clamp_(-self.epsilon, self.epsilon)

        return injection.detach()