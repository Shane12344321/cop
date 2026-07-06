You are acting as a Principal ML Research Scientist. I have provided the core files from DeepMind's searchless_chess repository. 

Our objective is to solve the "Dual-Capability Bottleneck" (the architectural tension where sequence tracking requires low-Elo data diversity, but policy decision quality requires high-Elo data filtering). We want to solve this algorithmically using a Dual-Head Multi-Task Architecture.

Please act as the architect and design a blueprint for the following modifications:

1. ARCHITECTURAL SHIFT (transformer.py): Redesign the decoder-only transformer trunk to split into two distinct output heads:
   - Head A (Tracking Head): Responsible for cross-entropy state/board tracking across the full sequence trajectory.
   - Head B (Policy/Value Head): Responsible for action-value (Q) or state-value (V) prediction.

2. MATHEMATICAL LOSS FORMULATION (training.py): Define a joint loss function:
   $$L_{total} = \alpha L_{tracking} + \beta(Elo) L_{deciding}$$
   Formulate a dynamic gradient scaling schedule where \beta is dynamically weighted based on the Elo rating of the training batch (allowing low-Elo games to contribute fully to state tracking while minimizing their influence on the policy/deciding head).

3. DATA FLOW INTEGRATION (data_loader.py): Outline the exact tensor shape transitions required to pipe the history tokens and Elo metadata simultaneously into these two heads without breaking JAX/Flax immutability.

CRITICAL CONSTRAINT: Do not write the full production code implementation yet. Provide a rigorous, step-by-step structural specification, mathematical formulations using LaTeX, and pseudo-code blocks showing exactly where the existing methods in transformer.py and training.py must be modified.
