# SOLOMON AGENT

**A Three-Tier Reasoning System for Complex Problem Solving**

Solomon Agent is an advanced orchestration framework built around a local model. It applies research-backed reasoning techniques—Solomon's Paradox self-distancing, least-to-most decomposition, multi-agent debate, and chain-of-verification—while strictly managing context windows so reasoning depth is never sacrificed to token limits.

---

## ⚠️ The Pitfalls Solomon Solves

Traditional LLM pipelines fail on complex tasks due to architectural blind spots. Solomon is explicitly designed to bypass these common failure modes:

*   **The "Conciseness" Trap:** Prompts demanding brevity cause models to truncate deep, complex answers. 
    *   *Fix:* Split system prompts (process-focused vs. answer-focused) and dedicated Final Synthesis phases.
*   **Reasoning Token Exhaustion:** Deep thinking consumes the entire 131K context window, resulting in empty outputs (`finish_reason="length"`). 
    *   *Fix:* The **Answer-Space Guard** strictly reserves 30K tokens for the final answer.
*   **The "Isolated Sub-Problem" Disaster:** Models solve parts of a problem in isolation but fail to assemble them, yielding one-line summaries. 
    *   *Fix:* A dedicated **Final Synthesis** phase explicitly forces integration.
*   **Short-Hard Questions:** 5-word prompts like *"how do i make digital life?"* break standard decomposition logic. 
    *   *Fix:* **Conceptual Depth Classification** and a **Question Expansion** step map out required domains before work begins.
*   **Vague Critique Thrashing:** Generic "revise" prompts cause the model to loop and shrink the answer. 
    *   *Fix:* **Directional Verdicts** (EXPAND, CORRECT, RESTRUCTURE, PASS) with specific sufficiency rules.

---

## 🏗️ Core Design Principles

| Principle | Description |
| :--- | :--- |
| **Process over Persona** | Tell the model *how* to work, not *who* to be. No personas, only processes. |
| **Generator ≠ Critic** | Critique always runs in a fresh, isolated context window to prevent bias. |
| **Fresh Phase Windows** | There is no global token budget. Each phase gets a full 131K context window. |
| **Reason Long, Carry Little** | Models may use 100K tokens thinking; only distilled conclusions move forward. |

---

## 🚀 The Three Tiers

Solomon dynamically routes tasks based on complexity scoring (0.0 to 1.0).

### ⚡ Lightning (Complexity < 0.3)
*For quick fixes, lookups, and boilerplate.*
**Pipeline:** Task → Reframe (third-person) → Solve → Fast Critique → Answer.

### 🧠 Deep (Complexity 0.3 – 0.7)
*For multi-step tasks, refactors, and debugging.*
**Pipeline:** Task → Decompose → Isolated Sub-problems → Multi-agent Debate (3 task-focused analysts) → Self-refine loop → Answer.

### 🌌 Omega (Complexity > 0.7)
*For novel problems, deep questions, and high-stakes work.*
**Pipeline:** Meta-prompt → ToT/Linear CoT → Adversarial Debate → Self-consistency → Chain-of-Verification → **Final Synthesis** → Adversarial Refine → Answer.

---

## 🛡️ Key Defensive Mechanisms

### 1. Answer-Space Guard
Prevents the model from thinking itself into a corner. If reasoning approaches 100K tokens without an answer, the guard injects a prompt to force the final answer. If the output truncates, the orchestrator retries with doubled `max_tokens`.

### 2. Type-Aware Compression Protocol
Full reasoning traces are stored for auditing but **never** passed to the next phase. An extraction step distills sub-problem outputs into JSON (Answer, Code, Insights, Rejected Approaches), achieving ~92% context reduction while preserving 100% of the conclusions.

### 3. Graceful "FINISH NOW"
A proactive abort button that doesn't just kill the process. It takes the partial pipeline state and runs a **separate, fully-budgeted synthesis call** to provide the best possible answer from the current context, complete with honest gap acknowledgments.

### 4. Short-Hard Question Expansion
Detects abstract, high-depth questions with low word counts. Before decomposition, it runs a Question Expansion step to map out the domains required, ensuring the model doesn't under-answer massive concepts.

---

## 📦 Implementation Status & Priorities

*   **Tier 1 (Core Stability):** Split system prompts, Answer-space guard, Answer extraction.
*   **Tier 2 (Depth & Quality):** Final Synthesis phase, Directional critique, Distilled debate inputs.
*   **Tier 3 (UX & Polish):** FINISH NOW, Trace Viewer, Confidence streaming.

---

*Built for OpenAI-compatible local endpoint | Context: 131K Tokens*
