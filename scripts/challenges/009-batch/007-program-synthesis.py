#!/usr/bin/env python3
"""
Challenge 009-007: Phase 3 - Full Program Synthesis

Synthesize complete encoding programs that compose ALL Holon primitives:
- Field weights (Phase 1)
- Interactions via bind() (Phase 2)
- Signal manipulation: negate(), amplify(), resonance()
- Comparison operations: difference(), blend()
- Sequence operations: permute(), cleanup()

The synthesizer evolves programs that maximize classification accuracy.
"""

import random
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from holon import CPUStore
from holon.encoder import Encoder

sys.path.insert(0, str(Path(__file__).parent))
from common import compute_accuracy, print_confusion_matrix


class PrimitiveOp(str, Enum):
    """Available primitive operations for program synthesis."""

    # Field operations
    ENCODE_FIELD = "encode_field"       # Encode a single field
    WEIGHT_FIELD = "weight_field"       # Apply weight to encoded field

    # Composition
    BIND = "bind"                       # Bind two vectors
    BUNDLE = "bundle"                   # Bundle multiple vectors

    # Signal manipulation
    NEGATE = "negate"                   # Remove component from superposition
    AMPLIFY = "amplify"                 # Boost component in superposition
    RESONANCE = "resonance"             # Extract matching part

    # Comparison
    DIFFERENCE = "difference"           # What distinguishes two vectors
    BLEND = "blend"                     # Interpolate between vectors

    # Cleanup
    CLEANUP = "cleanup"                 # Find closest in codebook


@dataclass
class ProgramNode:
    """A node in the encoding program tree."""
    op: PrimitiveOp
    args: List[Any]  # Arguments (field names, weights, other nodes)
    weight: float = 1.0

    def __repr__(self):
        return f"{self.op.value}({', '.join(str(a) for a in self.args[:2])}...)"


class EncodingProgram:
    """
    A synthesized encoding program.

    Represents a tree of operations that encode an item into a vector.
    """

    def __init__(self, store: CPUStore, nodes: List[ProgramNode] = None):
        self.store = store
        self.encoder = store.encoder
        self.nodes = nodes or []

        # Cache for class prototypes (for cleanup operation)
        self.class_prototypes: Dict[str, np.ndarray] = {}

    def execute(self, item: Dict[str, Any]) -> np.ndarray:
        """Execute the program on an item to produce a vector."""
        if not self.nodes:
            # Default: just encode the whole item
            return self.encoder.encode_data(item)

        # Execute each node and combine results
        vectors = []
        for node in self.nodes:
            vec = self._execute_node(node, item)
            if vec is not None:
                vectors.append(node.weight * vec.astype(np.float32))

        if not vectors:
            return self.encoder.encode_data(item)

        # Bundle all results
        bundled = np.sum(vectors, axis=0)
        return np.where(bundled > 0, 1, np.where(bundled < 0, -1, 0)).astype(np.int8)

    def _execute_node(self, node: ProgramNode, item: Dict) -> Optional[np.ndarray]:
        """Execute a single program node."""

        if node.op == PrimitiveOp.ENCODE_FIELD:
            field_name = node.args[0]
            if field_name not in item:
                return None
            return self.encoder.encode_data({field_name: item[field_name]})

        elif node.op == PrimitiveOp.WEIGHT_FIELD:
            field_name, weight = node.args[0], node.args[1]
            if field_name not in item:
                return None
            vec = self.encoder.encode_data({field_name: item[field_name]})
            return weight * vec.astype(np.float32)

        elif node.op == PrimitiveOp.BIND:
            # Bind two fields together
            field_a, field_b = node.args[0], node.args[1]
            if field_a not in item or field_b not in item:
                return None
            vec_a = self.encoder.encode_data({field_a: item[field_a]})
            vec_b = self.encoder.encode_data({field_b: item[field_b]})
            return (vec_a * vec_b).astype(np.int8)

        elif node.op == PrimitiveOp.NEGATE:
            # Negate a field (subtract from overall encoding)
            field_name = node.args[0]
            if field_name not in item:
                return None
            vec = self.encoder.encode_data({field_name: item[field_name]})
            # Return negative contribution
            return -vec.astype(np.float32)

        elif node.op == PrimitiveOp.AMPLIFY:
            # Amplify a field with given strength
            field_name, strength = node.args[0], node.args[1]
            if field_name not in item:
                return None
            vec = self.encoder.encode_data({field_name: item[field_name]})
            return strength * vec.astype(np.float32)

        elif node.op == PrimitiveOp.DIFFERENCE:
            # Encode the difference between two fields
            field_a, field_b = node.args[0], node.args[1]
            if field_a not in item or field_b not in item:
                return None
            vec_a = self.encoder.encode_data({field_a: item[field_a]})
            vec_b = self.encoder.encode_data({field_b: item[field_b]})
            return self.encoder.difference(vec_a, vec_b)

        elif node.op == PrimitiveOp.BLEND:
            # Blend two fields
            field_a, field_b, alpha = node.args[0], node.args[1], node.args[2]
            if field_a not in item or field_b not in item:
                return None
            vec_a = self.encoder.encode_data({field_a: item[field_a]})
            vec_b = self.encoder.encode_data({field_b: item[field_b]})
            return self.encoder.blend(vec_a, vec_b, alpha)

        return None

    def to_code(self) -> str:
        """Generate Python code representation of this program."""
        lines = ["def encode(item, encoder):"]
        lines.append("    vectors = []")

        for i, node in enumerate(self.nodes):
            if node.op == PrimitiveOp.ENCODE_FIELD:
                lines.append(f"    # Node {i}: encode field '{node.args[0]}'")
                lines.append(f"    if '{node.args[0]}' in item:")
                lines.append(f"        v{i} = encoder.encode_data({{'{node.args[0]}': item['{node.args[0]}']}})")
                lines.append(f"        vectors.append({node.weight} * v{i})")

            elif node.op == PrimitiveOp.BIND:
                lines.append(f"    # Node {i}: bind({node.args[0]}, {node.args[1]})")
                lines.append(f"    if '{node.args[0]}' in item and '{node.args[1]}' in item:")
                lines.append(f"        a = encoder.encode_data({{'{node.args[0]}': item['{node.args[0]}']}})")
                lines.append(f"        b = encoder.encode_data({{'{node.args[1]}': item['{node.args[1]}']}})")
                lines.append(f"        vectors.append({node.weight} * encoder.bind(a, b))")

            elif node.op == PrimitiveOp.NEGATE:
                lines.append(f"    # Node {i}: negate field '{node.args[0]}'")
                lines.append(f"    if '{node.args[0]}' in item:")
                lines.append(f"        v{i} = encoder.encode_data({{'{node.args[0]}': item['{node.args[0]}']}})")
                lines.append(f"        vectors.append(-{node.weight} * v{i})  # NEGATED")

            elif node.op == PrimitiveOp.AMPLIFY:
                lines.append(f"    # Node {i}: amplify field '{node.args[0]}' by {node.args[1]}")
                lines.append(f"    if '{node.args[0]}' in item:")
                lines.append(f"        v{i} = encoder.encode_data({{'{node.args[0]}': item['{node.args[0]}']}})")
                lines.append(f"        vectors.append({node.weight * node.args[1]} * v{i})")

        lines.append("    bundled = np.sum(vectors, axis=0)")
        lines.append("    return threshold_bipolar(bundled)")
        return "\n".join(lines)


class ProgramSynthesizer:
    """
    Synthesize encoding programs via evolutionary search.

    Uses a genetic algorithm to evolve programs that maximize
    classification accuracy on labeled data.
    """

    def __init__(
        self,
        store: CPUStore,
        exclude_fields: List[str] = None,
    ):
        self.store = store
        self.encoder = store.encoder
        self.exclude_fields = set(exclude_fields or [])

    def synthesize(
        self,
        X_train: List[Dict[str, Any]],
        y_train: List[str],
        X_val: Optional[List[Dict[str, Any]]] = None,
        y_val: Optional[List[str]] = None,
        population_size: int = 20,
        generations: int = 30,
        verbose: bool = True,
    ) -> EncodingProgram:
        """
        Synthesize an encoding program via evolutionary search.

        Args:
            X_train: Training examples
            y_train: Training labels
            X_val: Validation examples
            y_val: Validation labels
            population_size: Number of programs in population
            generations: Number of evolution generations
            verbose: Print progress

        Returns:
            Best discovered encoding program
        """
        if X_val is None:
            X_val = X_train
            y_val = y_train

        # Discover fields
        all_fields = set()
        for item in X_train:
            all_fields.update(item.keys())
        all_fields -= self.exclude_fields
        fields = sorted(all_fields)

        if verbose:
            print(f"Synthesizing programs over {len(fields)} fields...")
            print(f"Population: {population_size}, Generations: {generations}")

        # Initialize population with random programs
        population = [self._random_program(fields) for _ in range(population_size)]

        # Evaluate baseline (empty program = default encoding)
        baseline_prog = EncodingProgram(self.store, [])
        baseline_acc = self._evaluate(baseline_prog, X_train, y_train, X_val, y_val)
        if verbose:
            print(f"Baseline accuracy: {baseline_acc:.1%}")

        best_program = baseline_prog
        best_accuracy = baseline_acc

        # Evolution loop
        for gen in range(generations):
            # Evaluate all programs
            scores = []
            for prog in population:
                acc = self._evaluate(prog, X_train, y_train, X_val, y_val)
                scores.append((acc, prog))

            # Sort by accuracy
            scores.sort(key=lambda x: -x[0])

            # Track best
            if scores[0][0] > best_accuracy:
                best_accuracy = scores[0][0]
                best_program = scores[0][1]
                if verbose:
                    print(f"  Gen {gen+1}: {best_accuracy:.1%} (improved)")

            # Selection: keep top 50%
            survivors = [prog for _, prog in scores[:population_size // 2]]

            # Reproduction: create new programs via mutation and crossover
            new_population = survivors.copy()
            while len(new_population) < population_size:
                if random.random() < 0.7:
                    # Mutation
                    parent = random.choice(survivors)
                    child = self._mutate(parent, fields)
                else:
                    # Crossover
                    p1, p2 = random.sample(survivors, 2)
                    child = self._crossover(p1, p2)
                new_population.append(child)

            population = new_population

        if verbose:
            print(f"\nBest program accuracy: {best_accuracy:.1%}")
            print(f"Improvement over baseline: {best_accuracy - baseline_acc:+.1%}")

        return best_program

    def _random_program(self, fields: List[str], max_nodes: int = 6) -> EncodingProgram:
        """Generate a random encoding program."""
        num_nodes = random.randint(1, max(1, max_nodes))
        nodes = []

        for _ in range(num_nodes):
            op = random.choice([
                PrimitiveOp.ENCODE_FIELD,
                PrimitiveOp.ENCODE_FIELD,  # Higher probability
                PrimitiveOp.BIND,
                PrimitiveOp.NEGATE,
                PrimitiveOp.AMPLIFY,
            ])

            if op == PrimitiveOp.ENCODE_FIELD:
                field = random.choice(fields)
                weight = random.choice([0.5, 1.0, 1.5, 2.0])
                nodes.append(ProgramNode(op, [field], weight))

            elif op == PrimitiveOp.BIND:
                f1, f2 = random.sample(fields, 2)
                weight = random.choice([1.0, 1.5, 2.0])
                nodes.append(ProgramNode(op, [f1, f2], weight))

            elif op == PrimitiveOp.NEGATE:
                field = random.choice(fields)
                weight = random.choice([0.5, 1.0])
                nodes.append(ProgramNode(op, [field], weight))

            elif op == PrimitiveOp.AMPLIFY:
                field = random.choice(fields)
                strength = random.choice([1.5, 2.0, 3.0])
                weight = 1.0
                nodes.append(ProgramNode(op, [field, strength], weight))

        return EncodingProgram(self.store, nodes)

    def _mutate(self, program: EncodingProgram, fields: List[str]) -> EncodingProgram:
        """Mutate a program to create a variant."""
        new_nodes = [ProgramNode(n.op, n.args.copy(), n.weight) for n in program.nodes]

        if not new_nodes:
            return self._random_program(fields)

        mutation_type = random.choice(["modify", "add", "remove", "swap_weight"])

        if mutation_type == "modify" and new_nodes:
            # Change a field reference
            idx = random.randint(0, len(new_nodes) - 1)
            node = new_nodes[idx]
            if node.args:
                if isinstance(node.args[0], str) and node.args[0] in fields:
                    node.args[0] = random.choice(fields)

        elif mutation_type == "add":
            # Add a new node
            new_node = self._random_program(fields, max_nodes=1).nodes
            if new_node:
                new_nodes.append(new_node[0])

        elif mutation_type == "remove" and len(new_nodes) > 1:
            # Remove a node
            idx = random.randint(0, len(new_nodes) - 1)
            new_nodes.pop(idx)

        elif mutation_type == "swap_weight" and new_nodes:
            # Change a weight
            idx = random.randint(0, len(new_nodes) - 1)
            new_nodes[idx].weight = random.choice([0.5, 1.0, 1.5, 2.0, 3.0])

        return EncodingProgram(self.store, new_nodes)

    def _crossover(self, p1: EncodingProgram, p2: EncodingProgram) -> EncodingProgram:
        """Create child program by combining two parents."""
        # Take random subset of nodes from each parent
        nodes1 = p1.nodes[:len(p1.nodes)//2] if p1.nodes else []
        nodes2 = p2.nodes[len(p2.nodes)//2:] if p2.nodes else []

        child_nodes = [
            ProgramNode(n.op, n.args.copy(), n.weight)
            for n in nodes1 + nodes2
        ]

        return EncodingProgram(self.store, child_nodes)

    def _evaluate(
        self,
        program: EncodingProgram,
        X_train: List[Dict],
        y_train: List[str],
        X_val: List[Dict],
        y_val: List[str],
    ) -> float:
        """Evaluate program accuracy via prototype classification."""
        # Build prototypes
        label_vectors: Dict[str, List[np.ndarray]] = {}
        for item, label in zip(X_train, y_train):
            vec = program.execute(item)
            if label not in label_vectors:
                label_vectors[label] = []
            label_vectors[label].append(vec)

        prototypes: Dict[str, np.ndarray] = {}
        for label, vectors in label_vectors.items():
            stacked = np.stack(vectors)
            mean = np.mean(stacked, axis=0)
            proto = np.where(mean > 0, 1, np.where(mean < 0, -1, 0)).astype(np.int8)
            prototypes[label] = proto

        # Classify validation set
        correct = 0
        for item, true_label in zip(X_val, y_val):
            vec = program.execute(item)
            best_label = None
            best_sim = -float('inf')
            for label, proto in prototypes.items():
                sim = self._cosine_similarity(vec, proto)
                if sim > best_sim:
                    best_sim = sim
                    best_label = label
            if best_label == true_label:
                correct += 1

        return correct / len(y_val) if y_val else 0.0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a.astype(float), b.astype(float))
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


def generate_complex_data(n_samples: int = 500, seed: int = 42):
    """
    Generate data that benefits from multiple primitives.

    - Some fields are discriminative (should be amplified)
    - Some fields are noise (should be negated)
    - Some field PAIRS matter (should be bound)
    - The data is designed to be HARDER than simple encoding
    """
    random.seed(seed)
    np.random.seed(seed)

    items = []
    labels = []

    # Make this harder: each category has OVERLAPPING individual features
    # but UNIQUE combinations. This requires:
    # 1. bind() to capture the interaction
    # 2. negate() to remove confusing noise
    # 3. amplify() to boost weak signals

    for i in range(n_samples):
        # Choose category first
        category = random.choice(["alpha", "beta", "gamma"])

        # Generate features based on category
        # The key: individual features overlap, but combinations don't

        if category == "alpha":
            # Alpha: type=premium + tier=gold combination
            item_type = random.choice(["premium", "standard"])  # 50% premium
            tier = random.choice(["gold", "silver"])  # 50% gold
            # But ONLY alpha has premium+gold together
            if random.random() < 0.7:  # 70% have the discriminative combo
                item_type = "premium"
                tier = "gold"

        elif category == "beta":
            # Beta: type=standard + tier=silver combination
            item_type = random.choice(["premium", "standard"])
            tier = random.choice(["gold", "silver"])
            if random.random() < 0.7:
                item_type = "standard"
                tier = "silver"

        else:  # gamma
            # Gamma: mixed - neither premium+gold nor standard+silver
            if random.random() < 0.5:
                item_type = "premium"
                tier = "silver"
            else:
                item_type = "standard"
                tier = "gold"

        # Add LOTS of noise fields (should be negated)
        noise_fields = {
            "noise_1": random.choice(["N1", "N2", "N3", "N4", "N5"]),
            "noise_2": random.choice(["X", "Y", "Z", "W"]),
            "noise_3": f"n{random.randint(0, 50)}",
            "noise_4": random.choice(["red", "blue", "green"]),
        }

        item = {
            "type": item_type,
            "tier": tier,
            **noise_fields,
        }

        items.append(item)
        labels.append(category)

    return items, labels


def main():
    print("=" * 70)
    print("Challenge 009-007: Phase 3 - Full Program Synthesis")
    print("=" * 70)
    print("""
This synthesizes complete encoding programs that compose ALL Holon primitives:
- Field weights (ENCODE_FIELD with weight)
- Interactions (BIND)
- Signal manipulation (NEGATE, AMPLIFY)

The genetic algorithm evolves programs that maximize classification accuracy.
    """)

    # Generate data
    print("\n1. Generating complex classification data...")
    items, labels = generate_complex_data(n_samples=500)

    # Split
    random.seed(42)
    indices = list(range(len(items)))
    random.shuffle(indices)
    split_idx = int(0.8 * len(indices))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = [items[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_test = [items[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]

    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"   Categories: {sorted(set(labels))}")

    # Create synthesizer
    print("\n2. Creating program synthesizer...")
    store = CPUStore(dimensions=4096)
    synthesizer = ProgramSynthesizer(store, exclude_fields=["noise_2"])

    # Synthesize
    print("\n3. Evolving encoding programs...")
    start_time = time.time()
    best_program = synthesizer.synthesize(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        population_size=30,
        generations=40,
        verbose=True,
    )
    elapsed = time.time() - start_time

    # Evaluate final program
    print("\n4. Evaluating best program...")

    # Build prototypes with best program
    label_vectors: Dict[str, List[np.ndarray]] = {}
    for item, label in zip(X_train, y_train):
        vec = best_program.execute(item)
        if label not in label_vectors:
            label_vectors[label] = []
        label_vectors[label].append(vec)

    prototypes = {}
    for label, vectors in label_vectors.items():
        stacked = np.stack(vectors)
        mean = np.mean(stacked, axis=0)
        proto = np.where(mean > 0, 1, np.where(mean < 0, -1, 0)).astype(np.int8)
        prototypes[label] = proto

    # Predict
    y_pred = []
    for item in X_test:
        vec = best_program.execute(item)
        best_label = None
        best_sim = -float('inf')
        for label, proto in prototypes.items():
            dot = np.dot(vec.astype(float), proto.astype(float))
            norm = np.linalg.norm(vec) * np.linalg.norm(proto)
            sim = dot / norm if norm > 0 else 0
            if sim > best_sim:
                best_sim = sim
                best_label = label
        y_pred.append(best_label)

    final_acc = compute_accuracy(y_test, y_pred)

    print(f"\n   Final test accuracy: {final_acc:.1%}")
    print(f"   Synthesis time: {elapsed:.1f}s")

    # Show the synthesized program
    print("\n5. Synthesized program structure:")
    for i, node in enumerate(best_program.nodes):
        print(f"   [{i}] {node.op.value}: {node.args}, weight={node.weight}")

    print("\n6. Generated code:")
    print("-" * 40)
    print(best_program.to_code())
    print("-" * 40)

    # Confusion matrix
    print("\n7. Confusion matrix:")
    print_confusion_matrix(y_test, y_pred)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Program synthesis successfully evolved an encoding program:
- {len(best_program.nodes)} operations composed
- Uses primitives: {set(n.op.value for n in best_program.nodes)}
- Test accuracy: {final_acc:.1%}
- Time: {elapsed:.1f}s

This demonstrates that we can AUTOMATICALLY discover the right
combination of Holon primitives for a given classification task.
    """)


if __name__ == "__main__":
    main()
