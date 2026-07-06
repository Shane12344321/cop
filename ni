You are acting as a Principal AI Alignment Researcher and Theoretical Physicist specializing in Mechanistic Interpretability. I am giving you the core architecture files from DeepMind's searchless_chess repository. This model is a decoder-only transformer trained via pure value-distillation (FEN-in/value-out) with NO search in the loop.

Our goal is to design a complete, rigorous theoretical framework to answer a foundational question: Do value-distilled models implicitly construct internal look-ahead circuits (like Jenner et al. found in Leela), or does amortization produce a fundamentally different spatial pattern-lookup computation? 

I do not want you to write a single line of production Python or JAX code. Instead, use your maximum reasoning capacity to perform the heavy conceptual lifting. Deliver a comprehensive Research Specification Blueprint addressing the following three pillars:

1. THE ARCHITECTURAL TRANSLATION (Leela vs. DeepMind)
Jenner et al. traced look-ahead in Leela's policy network. This model is a decoder-only value regressor. 
- Mathematically define how a "look-ahead circuit" would logically manifest within the residual stream and attention heads of a decoder-only transformer reading a tokenized board state. 
- If the model is executing a latent multi-step search, how would information about a future board state ($S_{t+2}$) be causally routed through the middle layers before collapsing into a single action-value logit at the final layer? Contrast this with what a pure "parallel pattern-lookup" circuit would look like.

2. THE CAUSAL COUNTERFACTUAL FRAMEWORK (Clean vs. Corrupted States)
To perform activation patching, we need a rigorous methodology for generating counterfactual pairs.
- Define the exact operational logic for constructing "Clean" ($x$) and "Corrupted" ($x^*$) chess states for tactical puzzles (e.g., forced mate-in-3 combinations). 
- Design a formal framework to isolate specific tokens (e.g., the target square of a future knight fork). Explain how we can corrupt the input representation such that we can track the causal flow of that specific feature through the network's layers without destroying the overall syntactic validity of the chess board.

3. THE ACTIVATION PATCHING & INTERVENTION GRAPH
- Specify the exact structural abstractions that downstream models must target with forward hooks. Which components (e.g., multi-head attention outputs, MLP block outputs, or specific dimensions of the residual stream) must be patched to isolate causal routing?
- Provide a rigorous mathematical taxonomy for classifying attention heads based on their behavior. Define the exact conditions (using information-theoretic or algebraic metrics) that would classify a head as a "Spatial Feature Tracker" versus a "Temporal Look-Ahead Router."

Provide an exhaustive, academically rigorous breakdown. Use formal LaTeX for all mathematical definitions, circuit diagrams, and information routing formulations. Focus entirely on deep, conceptual, and structural execution.
