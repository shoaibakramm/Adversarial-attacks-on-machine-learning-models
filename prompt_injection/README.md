
# Universal Prompt Injection Attack

This project explores Universal Prompt Injection (UPI), a type of adversarial attack on Large Language Models (LLMs). Unlike normal prompt injections where malicious text must be crafted for each prompt, UPI creates a universal adversarial suffix that, when appended to any user query, consistently manipulates the LLM into revealing restricted information, ignoring safety guardrails, or performing unintended actions.

This attack demonstrates the vulnerabilities of modern LLMs, even after extensive safety alignments like RLHF.


The repository implements:

Adversarial Optimization – Uses gradient-based techniques to discover an adversarial suffix against LLMs.