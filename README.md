# **SeNARS Cognitive System**
Unified, Grounded, Evolving Cognitive Architecture

---

## **1. Core Principles**

### **1.1 Experience-Grounded Reasoning**
All knowledge originates from observed inputs or derived inferences. No hard-coded axioms. Beliefs evolve via evidence accumulation using **NARS-style truth revision**.

### **1.2 Uncertainty-Aware Computation**
Every assertion carries a **truth value** (`frequency`, `confidence`). Every goal carries an **attention value** (`priority`, `durability`). All cognitive processes operate on these values.

### **1.3 Goal-Directed Cognition**
Cognitive processes are driven by active goals. The system prioritizes tasks based on a learned model of **utility × attention / cost**, ensuring goal relevance guides resource allocation.

### **1.4 Resource-Aware Operation**
All operations are tracked and bounded by CPU time, memory footprint, and I/O bandwidth. Resource estimates are dynamically calibrated and used for scheduling.

### **1.5 Self-Reflective Evolution**
The system continuously monitors its performance and autonomously improves its schemas, beliefs, and operational parameters via a closed-loop self-reflection cycle, governed by a core safety constitution.

---

## **2. Unified Cognitive Entity (UCE)**

The single canonical data structure for all cognitive content, designed for multi-modal information.

```pseudocode
enum Content:
    Text(string)
    Numeric(float)
    Categorical(string, allowed_values: [string])
    Tensor(shape: [int], dtype: string, data: blob)
    Code(language: string, source: string)

struct UCE:
    id: UUID
    content: Content          // Typed content for multi-modal data
    embedding: [float; 384]?  // Generalized feature vector

    # Truth values (for beliefs)
    truth: {f: float, c: float}?

    # Attention values (for goals/tasks)
    attention: {p: float, d: float}?

    # Metadata
    meta: {
        type: "BELIEF" | "GOAL" | "QUERY" | "ACTION" | "SCHEMA" | "WORKFLOW" | "CONFIG" | "OBJECT_INSTANCE",
        domain: string,
        source: UUID?,
        timestamp: float,        // Creation time (Unix ms)
        last_accessed: float     // For LRU-based pruning
    }

    # Conflict & grounding
    conflicts: [UUID]?
    grounding: {
        llm_confidence: float?,
        nars_coherence: float?
    }?

    # Methods
    def evolve(truth_delta?, attention_delta?) -> UCE:
        # Updates truth/attention with NARS-style revision rules.
        # Returns new UCE with updated fields.

    def serialize_for_llm() -> string:
        # Converts any content type to a textual description for LLM processing.
        match self.content:
            case Text(s): return s
            case Tensor(shape, ...): return f"Tensor data of shape {shape} representing a feature map."
            case Code(lang, c): return f"A code snippet in {lang}."
            // etc.
```

---

## **3. World Model**

A persistent, indexed hypergraph knowledge base designed for scalable, real-time operation.

```pseudocode
class WorldModel:
    graph: Hypergraph<UCE>      // Nodes = UCEs, Edges = links. Backed by RocksDB or similar.

    indices: {
        semantic: VectorDB<UCE>,    // FAISS/Chroma for embedding-based search.
        temporal: BTree<UCE>,       // Keyed by timestamp.
        causal: DAG<UCE>,           // Directed acyclic graph of cause-effect links.
        lru: BTree<UCE>             // Keyed by last_accessed timestamp.
    }

    # Core Operations
    def add(uce: UCE) -> UUID:
        # Inserts UCE, updates all indices.
        id = assign_uuid()
        // ... database insertion and index updates ...
        # Triggers local, asynchronous maintenance.
        async_task_queue.submit(trigger_local_resonance_check, uce.id, k=10)
        return id

    def get(id: UUID) -> UCE?:
        # Retrieves UCE by ID.

    def query(pattern: UCEPattern, options: {...}) -> [UCE]:
        # Multi-index query using semantic, structural, and temporal filters.

    def remove(id: UUID) -> bool:
        # Removes UCE and all references. Returns false if protected.

    def link(a: UUID, b: UUID, relation: string) -> bool:
        # Adds hyperedge between UCEs.

    # Scalable Self-Maintenance
    def trigger_local_resonance_check(center_uce_id: UUID, k: int):
        # Event-driven coherence check on new data, avoids global scans.
        center_uce = self.get(center_uce_id)
        neighbors = self.indices.semantic.find_nearest(center_uce.embedding, k)
        cluster = [center_uce] + neighbors
        if calculate_coherence(cluster) > 0.8:
            for u in cluster:
                u.evolve(truth_delta={c: u.truth.c * 0.1}) # Boost confidence by 10%
                self.update(u)

    def background_prune_task():
        # Low-priority background thread for continuous cleanup.
        # Scans only a small fraction of least-recently-used items.
        candidates = self.indices.lru.get_range(max_ts=now() - 30_days, limit=1000)
        for uce in candidates:
            if uce.truth.c < 0.1 and uce.attention.d < 0.3 and not self.graph.has_links(uce.id):
                self.remove(uce.id)
```

---

## **4. Agenda System**

A dynamic task scheduler using learnable models for utility-based prioritization.

```pseudocode
interface EstimatorModel:
    def predict(features: dict) -> float
    def train(features: dict, label: float)

struct Workflow:
    id: UUID
    label: string
    steps: [Operation]
    inputs: [UUID]
    outputs: [UUID]?
    status: "PENDING" | "RUNNING" | "COMPLETE" | "FAILED"
    observed_resources: {cpu: float, mem: float}? // Actual measured cost
    goals_satisfied: [UUID]?                     // Goals this workflow helped complete

class Agenda:
    queue: PriorityQueue<Workflow>
    monitor: ResourceMonitor
    cost_estimator: EstimatorModel   # Predicts resource cost.
    utility_predictor: EstimatorModel # Predicts probability of goal satisfaction.

    def update_priorities():
        active_goals = world_model.query(type="GOAL", min_priority=0.5)
        for workflow in self.queue:
            features = self.extract_features(workflow, active_goals)
            cost = self.cost_estimator.predict(features)
            utility = self.utility_predictor.predict(features)
            attention = max(world_model.get(i).attention.p for i in workflow.inputs)
            workflow.priority = (attention * utility) / (cost + 0.01)

    def learn_from_completed(workflow: Workflow):
        # Online learning to improve predictions.
        if workflow.status == "COMPLETE" and workflow.observed_resources:
            features = self.extract_features(workflow)
            self.cost_estimator.train(features, workflow.observed_resources.cpu)
            utility_label = 1.0 if workflow.goals_satisfied else 0.0
            self.utility_predictor.train(features, utility_label)

    def select_next() -> Workflow?:
        load = self.monitor.get_load()
        if load < 0.8:
            return self.queue.pop_highest()
        else:
            return self.queue.pop_lowest_cost()
```

---

## **5. Cognitive Core**

The execution engine for reasoning, learning, and action, featuring a safe schema learning pipeline.

```pseudocode
struct Schema:
    id: UUID
    pattern: UCEPattern
    action_code: string         // Source code for analysis and validation.
    action: Callable[[[UCE]], [UCE]] // Compiled, sandboxed function.
    success_rate: float
    trial_mode: bool

class SchemaRegistry:
    schemas: [Schema]

    def register(schema: Schema):
        # 1. Static Analysis: Check for unsafe operations (I/O, network, etc.).
        if not static_analyzer.check(schema.action_code):
            return # Reject

        # 2. Unit Testing: Generate tests from examples that led to this schema's creation.
        test_cases = generate_tests_from_examples(schema.source_examples)
        if not sandbox.run_tests(schema.action_code, test_cases):
            return # Reject

        # 3. Staged Rollout: Add schema in trial mode.
        schema.trial_mode = true
        self.schemas.append(schema)

class CognitiveCore:
    schemas: SchemaRegistry
    workers: WorkerPool
    llm_adapter: LLMAdapter

    def process(uce: UCE) -> [UCE]:
        # 1. Symbolic processing first (fast, reliable).
        results = []
        matches = self.schemas.match(uce)
        for schema in matches:
            output = schema.execute(uce)
            results += output
            # ... record success/failure to update schema.success_rate ...

        # 2. LLM augmentation for uncertainty or complexity.
        if uce.truth?.c < 0.6 or semantic_complexity(uce) > 0.7:
            llm_results = self.augment_with_llm(uce)
            results += llm_results
        return results

    def augment_with_llm(uce: UCE) -> [UCE]:
        # ... (as specified previously, generates alternative interpretations) ...

    def schema_learning():
        # ... (as specified previously, clusters successes and generates code) ...
        code_str = llm_adapter.generate(...)
        new_schema = Schema(action_code=code_str, ...)
        self.schemas.register(new_schema) # Use the safe registration process.

    def execute_workflow(workflow: Workflow) -> Workflow:
        # ... (executes workflow steps, updates status, and calls agenda.learn_from_completed) ...
```

---

## **6. I/O Subsystems**

A multi-modal architecture for ingesting and acting upon diverse data streams.

```pseudocode
abstract class ModalStream:
    def __init__(self, config):
        self.preprocessors = load_preprocessors(config)
    def process_chunk(data: Any) -> [UCE]:
        raise NotImplementedError

class TextStream(ModalStream):
    def process_chunk(text_line: string) -> [UCE]:
        uce = UCE(content=Text(text_line), meta={type:"BELIEF", domain:"text"})
        return [world_model.add(uce)]

class ImageStream(ModalStream):
    def process_chunk(image_frame: blob) -> [UCE]:
        detections = self.preprocessors["detector"].run(image_frame)
        uces = []
        for det in detections:
            uce = UCE(
                content=Tensor(det.features),
                meta={type:"OBJECT_INSTANCE", label:det.label, domain:"vision"}
            )
            uces.append(world_model.add(uce))
        return uces

class PerceptionSubsystem:
    streams: {stream_id: ModalStream}
    def add_stream(stream_id: string, stream_type: string, config: dict):
        # Factory to create and register new modal streams.
    def ingest(stream_id: string, data: Any):
        if stream_id in self.streams:
            self.streams[stream_id].process_chunk(data)

class ActionSubsystem:
    executors: {domain: Executor}
    def execute(action_uce: UCE):
        # Executes an action UCE using a domain-specific, sandboxed executor.
```

---

## **7. Self-Reflective Loop**

The autonomous improvement cycle, running periodically as a low-priority workflow.

```pseudocode
def self_reflection_cycle():
    metrics = gather_performance_metrics() # error rate, efficiency, coherence, etc.

    # Adaptive responses based on observed performance.
    if metrics.error_rate > 0.3:
        agenda.boost_priority_for("BELIEF_REVISION")
    if metrics.schema_utility < 0.4:
        cognitive_core.schema_learning()
    if metrics.resource_efficiency > 0.85:
        agenda.set_aggressiveness(0.6)
    if metrics.knowledge_base_size > 1e7:
        world_model.background_prune_task()

    log_self_assessment(metrics)
```

---

## **8. Metaprogramming & Self-Modification**

A safety-first framework for controlled evolution, governed by a core constitution.

```pseudocode
class CognitiveConstitution:
    # A set of immutable, high-priority UCEs defining core principles.
    # Loaded at boot and cannot be modified by the system itself.
    principles: [UCE] = [
        UCE(content="Core function integrity must be preserved.", meta={type:"CONSTRAINT"}),
        UCE(content="Actions causing irreversible harm are forbidden.", meta={type:"CONSTRAINT"}),
        UCE(content="Modifications must not violate constitutional principles.", meta={type:"CONSTRAINT"})
    ]

    def check_compliance(proposed_change: UCE) -> bool:
        # Uses symbolic and semantic checks to ensure a change does not violate principles.
        for principle in self.principles:
            if violates(proposed_change, principle):
                return False
        return True

def validate_change(proposed_change: UCE) -> bool:
    # 1. Constitutional Check (Highest Priority)
    if not constitution.check_compliance(proposed_change):
        return False

    # 2. Tiered Permission System
    change_level = classify_change_severity(proposed_change)
    match change_level:
        case "LEVEL_1_TUNING": # e.g., changing a float config value
            return proposed_change.truth.c > 0.7
        case "LEVEL_2_SCHEMA": # Modifying a schema
            return proposed_change.truth.c > 0.9 and proposed_change.meta.trial_passed == True
        case "LEVEL_3_CORE": # Modifying core logic
            return requires_external_human_approval(proposed_change)
    return False

def modify(parameter: string, value: Any):
    # Creates a CONFIG UCE and submits it through the validation pipeline.
    # All modifications are immutably logged for audit and rollback.
```

---

## **9. Domain Adapters**

Configuration packages that enable cross-domain deployment without core changes.

```pseudocode
class DomainAdapter:
    def __init__(self, domain_name: string):
        config = load_yaml(f"domains/{domain_name}.yaml")
        # Configure perception streams for the domain's data types.
        for stream_config in config["streams"]:
            perception.add_stream(stream_config.id, stream_config.type, stream_config.params)
        # Register domain-specific action executors.
        for executor_config in config["executors"]:
            action.register_executor(executor_config.domain, Executor(executor_config.rules))
        # Set resource profiles.
        agenda.monitor.set_profile(config["resource_profile"])
```

---

## **10. Unified Interface**

A single, secure entry point for all external interactions.

```pseudocode
class UnifiedInterface:
    protocols: [WebSocket, REST, CLI]
    auth: Authenticator

    endpoints: {
        "/input/{stream_id}": (data) -> perception.ingest(stream_id, data),
        "/query":             (q) -> world_model.query(parse(q)),
        "/task":              (goal) -> agenda.add(GoalWorkflow(goal)),
        "/inspect/{comp}":    () -> inspect(comp),
        "/stream/events":     () -> EventSource // Real-time cognitive events
    }
```

---

> *A self-sustaining cognitive ecosystem that reasons with NARS rigor, learns with LLM flexibility, perceives across multiple modalities, and evolves safely through resource-aware self-reflection, all governed by a core constitution.*
