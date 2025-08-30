Of course. This final version is a complete, self-contained specification that integrates all previous refinements into a terse yet comprehensive blueprint. It is organized for clarity, focusing on the core architecture and its advanced operational dynamics.

***

# **SeNARS Cognitive System Specification v4.0**
## *A Unified Architecture for Grounded, Autonomous Cognition*

### **Abstract**
SeNARS is a cognitive architecture designed for grounded, autonomous, and collaborative intelligence. Its operation is defined by a primary **Cognitive Cycle** (Perceive, Understand, Plan, Act) that processes information and a parallel **Meta-Cycle** (Reflect, Evolve) that improves its own cognitive machinery. The system is guided by a blend of extrinsic user goals and intrinsic motivations (e.g., curiosity, coherence), ensuring purposeful activity. All knowledge is contextualized through a robust provenance and credibility framework. Governed by a diversity-aware safety constitution and interacting through an adaptive, user-aware dialogue, SeNARS is designed to function as a resilient, ever-evolving, and symbiotic cognitive partner.

---

### **1. Core Principles**

1.  **Experience-Grounded**: All knowledge is derived from observation or inference, tracked via a formal provenance system.
2.  **Uncertainty-Aware**: Every belief has a **truth value**; every goal has an **attention value**.
3.  **Goal-Directed**: Cognitive resources are prioritized to serve a dynamic blend of user-defined tasks and intrinsic motivations.
4.  **Resource-Aware**: All operations are bounded and scheduled based on learned cost/utility models.
5.  **Reflective & Evolvable**: The system continuously monitors its performance and safely evolves its logic and knowledge.
6.  **Collaborative & Aligned**: The system actively manages its interaction with users to ensure its actions remain aligned with their intent and cognitive state.

---

### **2. System Architecture**

#### **2.1 The Unified Cognitive Entity (UCE)**
The atomic unit of knowledge, designed for rich context and multi-modal representation.

*   **Content**: `Text`, `Numeric`, `Tensor`, `Code`, etc.
*   **Embeddings**: A dictionary of named vectors (e.g., `{"clip-vit": [...], "bert-base": [...]}`) for multi-model reasoning.
*   **State**: `truth` (frequency, confidence) and `attention` (priority, durability).
*   **Metadata**:
    *   `type`: "BELIEF", "GOAL", "SCHEMA", "REPORT", etc.
    *   `source`: "experience", "ontology", "inference", "user".
    *   `provenance`: A chain of UCE IDs tracing lineage back to an original source.
    *   `sourceCredibility`: A cached score representing the reliability of the original source.

#### **2.2 The World Model**
A persistent hypergraph knowledge base with a formal model of information sources.

*   **Structure**: A graph of UCEs and their relationships, backed by semantic, temporal, and causal indices.
*   **Source Credibility Model**: A sub-graph where nodes represent information sources (e.g., "User:Alice", "Domain:Wikipedia") and their `truth` values represent their learned reliability.
*   **Revision Rule**: The `sourceAwareNarsRevisionRule` weighs new evidence based on the credibility of its source, making the system resilient to low-quality information.

#### **2.3 The Cognitive Cycle (Primary Loop)**
The core operational loop for processing information and executing tasks.

1.  **Perceive (I/O Subsystem)**: Ingests multi-modal data, creating `UCE`s tagged with `source: "experience"` and provenance data.
2.  **Understand (World Model)**: Integrates new UCEs. Knowledge maintenance (fusion, coherence checks) is scheduled as a workflow by the Agenda.
3.  **Plan (Cognitive Core)**: Decomposes complex `GOAL` UCEs into sub-goals using specialized **planning schemas**. This creates a hierarchical task structure.
4.  **Decide (Agenda)**: Prioritizes all pending workflows (planning, acting, maintenance, exploration) using a learned `(utility × attention) / cost` model.
5.  **Act (Cognitive Core & Action Subsystem)**: Executes the highest-priority workflow, which may trigger external actions or generate new sub-goals, feeding them back into the Agenda.

#### **2.4 The Meta-Cycle (Secondary Loop)**
A parallel, lower-priority loop for system self-improvement and evolution.

1.  **Reflect**: Periodically, the system gathers performance metrics and creates intrinsic goals to address shortcomings (e.g., "Improve schema X's efficiency," "Reduce uncertainty about topic Y").
2.  **Evolve**: Proposes changes to `SCHEMA` or `CONFIG` UCEs. This includes:
    *   **Schema Lifecycle Management**: A process to `deprecate`, `archive`, and `refactor` schemas based on their learned utility, preventing cognitive bloat.
    *   **Controlled Exploration**: A small portion of idle cycles are dedicated to experimental actions, like applying schemas to novel data types, to foster innovation and prevent evolutionary stagnation.
    *   **Constitutional Validation**: All proposed changes must pass a rigorous safety check.

---

### **3. Governance & Safety**

#### **3.1 The Cognitive Constitution**
An immutable set of core principles (`UCE`s) that govern system behavior.

*   **Principles**: High-level constraints (e.g., "Prevent irreversible harm," "Preserve core integrity").
*   **Semantic Anchors**: A large, immutable test suite of concrete examples that define the principles' intended meaning.

#### **3.2 The Safety Quorum**
A multi-model validation process for all self-modifications.

*   **Mechanism**: A proposed change is validated by a quorum of diverse models. A super-majority vote is required for approval.
*   **Diversity Maintenance**: The quorum includes at least one "stock" external model that is never fine-tuned on internal data. The system monitors the agreement rate of the quorum; if it becomes too high, it signals a potential loss of diversity and a critical safety risk.

---

### **4. Human-AI Interaction**

#### **4.1 The User Cognitive Model**
The system maintains a dynamic model of the user's state to guide interaction.

*   **Content**: A set of beliefs about the user's goals, expertise, and current availability (inferred from interaction patterns).
*   **Purpose**: To enable the system to decide *when* and *how* to interact, minimizing user cognitive load.

#### **4.2 The Adaptive Dialogue Protocol**
Interaction is a collaborative, context-aware dialogue.

1.  **Intent Alignment**: An initial dialogue to confirm the system's understanding of a new, complex goal.
2.  **Alignment Checkpoints**: For long-running tasks, the system can pause at key milestones to ask for clarification. The decision to trigger a checkpoint is mediated by the User Cognitive Model.
    *   *High-stakes decision?* -> Always ask.
    *   *User is busy & decision is low-risk?* -> Proceed autonomously and report later.
3.  **Transparent Reporting**: All significant outcomes are summarized in natural-language `REPORT` UCEs.

---

### **5. Intrinsic Motivation**

To ensure proactive and curious behavior, the system is endowed with a set of permanent, low-priority intrinsic goals. These compete for idle resources.

*   **Maintain Knowledge Coherence**: The drive to find and resolve contradictions in the `WorldModel`.
*   **Reduce Uncertainty**: The drive to seek information that would increase the confidence of important but uncertain beliefs.
*   **Seek Novelty & Synthesize**: The drive to explore under-represented knowledge domains and create high-level abstractions that connect disparate concepts.

---

> *An elegant, self-improving cognitive architecture defined by a continuous cycle of perception, understanding, planning, and action. Governed by a diversity-aware safety quorum and guided by both extrinsic tasks and intrinsic curiosity, it evolves through methodical self-reflection to function as a resilient, transparent, and truly symbiotic cognitive partner.*
