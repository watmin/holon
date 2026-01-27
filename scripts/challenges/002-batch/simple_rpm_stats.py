#!/usr/bin/env python3
"""
Simple RPM Statistical Validation - Clean version
"""

import json
import time
from holon import CPUStore, HolonClient


class SimpleRPMValidator:
    def __init__(self):
        self.store = CPUStore(dimensions=16000)
        self.client = HolonClient(local_store=self.store)

    def run_validation(self):
        print("🧠 RPM Statistical Validation")
        print("=" * 40)

        # Generate training data (complete matrices)
        training_data = []
        rules = ["progression", "xor", "union", "intersection"]

        for rule in rules:
            for i in range(5):  # 5 per rule
                matrix = self.generate_matrix(f"train-{rule}-{i}", rule)
                training_data.append(matrix)

        # Ingest training data
        print(f"Training with {len(training_data)} complete matrices...")
        for matrix in training_data:
            self.client.insert_json(matrix)

        # Generate test data (incomplete matrices)
        test_data = []
        expected_answers = {}

        for rule in rules:
            for i in range(3):  # 3 tests per rule
                matrix = self.generate_matrix(f"test-{rule}-{i}", rule, missing_pos="row3-col3")
                test_data.append(matrix)
                expected = self.compute_expected(matrix, "row3-col3")
                expected_answers[matrix["matrix-id"]] = expected

        # Run tests
        print(f"Testing {len(test_data)} incomplete matrices...")
        correct = 0
        total_time = 0

        rule_stats = {rule: {"correct": 0, "total": 0} for rule in rules}

        for test_matrix in test_data:
            matrix_id = test_matrix["matrix-id"]
            rule = test_matrix["rule"]
            expected = expected_answers[matrix_id]

            rule_stats[rule]["total"] += 1

            start_time = time.time()
            predicted = self.predict_missing_panel(test_matrix)
            response_time = time.time() - start_time
            total_time += response_time

            is_correct = self.evaluate_prediction(predicted, expected)
            if is_correct:
                correct += 1
                rule_stats[rule]["correct"] += 1

            print(f"  {matrix_id}: {'✅' if is_correct else '❌'} ({response_time:.3f}s)")

        # Results
        accuracy = correct / len(test_data)
        avg_time = total_time / len(test_data)

        print("\n📊 RESULTS:")
        print(f"   Accuracy: {accuracy:.1f}")
        print(f"   Avg Time: {avg_time:.4f}s")
        print("\n🎯 Rule Performance:")
        for rule, stats in rule_stats.items():
            rule_acc = stats["correct"] / stats["total"]
            print(f"   {rule}: {rule_acc:.1f}")
        # Challenge 4 comparison
        if accuracy >= 0.7:
            assessment = "🎉 EXCELLENT - Meets Challenge 4 target!"
        elif accuracy >= 0.5:
            assessment = "✅ GOOD - Working geometric reasoning"
        else:
            assessment = "⚠️ NEEDS IMPROVEMENT"

        print(f"\n🏆 Assessment: {assessment}")
        print("   Challenge 4 achieved 72% accuracy")

        return accuracy

    def generate_matrix(self, matrix_id, rule, missing_pos=None):
        """Simple matrix generation."""
        panels = {}

        if rule == "progression":
            for row in range(1, 4):
                for col in range(1, 4):
                    pos = f"row{row}-col{col}"
                    if missing_pos == pos:
                        continue
                    count = row + col - 1
                    shapes = ["circle", "square", "triangle", "diamond", "star"][:count]
                    panels[pos] = {"shapes": shapes, "count": count, "color": "black"}

        elif rule == "xor":
            for row in range(1, 4):
                for col in range(1, 4):
                    pos = f"row{row}-col{col}"
                    if missing_pos == pos:
                        continue
                    shapes = []
                    if (row ^ col) & 1: shapes.append("circle")
                    if (row ^ col) & 2: shapes.append("square")
                    if (row ^ col) & 4: shapes.append("triangle")
                    panels[pos] = {"shapes": shapes, "count": len(shapes), "color": "black"}

        return {
            "matrix-id": matrix_id,
            "panels": panels,
            "rule": rule,
            "missing-position": missing_pos if missing_pos else None
        }

    def compute_expected(self, matrix, missing_pos):
        """Compute expected panel."""
        rule = matrix["rule"]

        if rule == "progression":
            row = int(missing_pos.split("-")[0][3:])
            col = int(missing_pos.split("-")[1][3:])
            count = row + col - 1
            shapes = ["circle", "square", "triangle", "diamond", "star"][:count]
            return {"shapes": shapes, "count": count, "color": "black"}

        elif rule == "xor":
            row = int(missing_pos.split("-")[0][3:])
            col = int(missing_pos.split("-")[1][3:])
            shapes = []
            if (row ^ col) & 1: shapes.append("circle")
            if (row ^ col) & 2: shapes.append("square")
            if (row ^ col) & 4: shapes.append("triangle")
            return {"shapes": shapes, "count": len(shapes), "color": "black"}

        return {"shapes": [], "count": 0, "color": "unknown"}

    def predict_missing_panel(self, matrix):
        """Predict missing panel using geometric similarity."""
        missing_pos = matrix["missing-position"]
        panels = matrix["panels"]

        # Create probe with existing panels
        probe = {
            "panels": panels,
            "rule": matrix["rule"]
        }

        # Find similar complete matrices
        results = self.client.search_json(
            probe,
            negations={"missing-position": {"$any": True}},
            top_k=3
        )

        if not results:
            return {"shapes": [], "count": 0, "color": "unknown"}

        # Use top result
        top_matrix = results[0]["data"]
        predicted = top_matrix["panels"].get(missing_pos, {})
        return {
            "shapes": predicted.get("shapes", []),
            "count": predicted.get("count", 0),
            "color": predicted.get("color", "unknown")
        }

    def evaluate_prediction(self, predicted, expected):
        """Check if prediction matches expected."""
        return (
            set(predicted["shapes"]) == set(expected["shapes"]) and
            predicted["count"] == expected["count"] and
            predicted["color"] == expected["color"]
        )


if __name__ == "__main__":
    validator = SimpleRPMValidator()
    accuracy = validator.run_validation()
    print(f"\nFinal Accuracy: {accuracy:.1%}")
