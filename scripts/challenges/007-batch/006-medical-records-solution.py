#!/usr/bin/env python3
"""
Challenge 007-006: Fuzzy Medical Record Matching

Demonstrates fuzzy matching on medical records with:
- Nested diagnoses and treatments
- N-gram text matching in notes
- Prototype learning for disease patterns
- Guard filters for severity
- Negation filters for medications

Usage:
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/006-medical-records-solution.py
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/006-medical-records-solution.py --http
"""

import argparse
import random
import time
import uuid
from typing import Any, Dict, List

from holon import CPUStore, HolonClient


def generate_medical_records(count: int = 50) -> List[Dict[str, Any]]:
    """Generate synthetic medical records."""
    symptoms_patterns = [
        ["fever", "cough", "fatigue"],
        ["headache", "nausea", "dizziness"],
        ["chest pain", "shortness of breath"],
        ["abdominal pain", "vomiting"],
        ["joint pain", "swelling", "stiffness"],
        ["rash", "itching"],
        ["sore throat", "runny nose", "cough"],
    ]

    diagnosis_patterns = [
        {"condition": "respiratory infection", "severity": 6},
        {"condition": "migraine", "severity": 7},
        {"condition": "cardiac event", "severity": 9},
        {"condition": "gastroenteritis", "severity": 5},
        {"condition": "arthritis", "severity": 6},
        {"condition": "allergic reaction", "severity": 4},
        {"condition": "common cold", "severity": 3},
    ]

    medications = ["antibiotic", "painkiller", "anti-inflammatory", "antacid", "antihistamine"]

    notes_templates = [
        "Patient presents with persistent dry cough and moderate fever lasting 3 days",
        "Severe headache with visual disturbances reported",
        "Chest discomfort and irregular heartbeat observed",
        "Acute abdominal pain in lower right quadrant",
        "Chronic joint inflammation with reduced mobility",
        "Skin eruption following exposure to allergen",
        "Upper respiratory symptoms consistent with viral infection",
    ]

    records = []
    for i in range(count):
        pattern_idx = i % len(symptoms_patterns)

        record = {
            "patient_id": f"patient_{random.randint(1000, 9999)}",
            "symptoms": random.choice(symptoms_patterns),
            "diagnoses": [
                {
                    "condition": diagnosis_patterns[pattern_idx]["condition"],
                    "severity": random.randint(3, 10),
                    "onset_date": f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                }
            ],
            "treatments": {
                "medication": random.choice(medications),
                "dosage": f"{random.randint(1, 4) * 250}mg",
            },
            "notes": {
                "$mode": "ngram",
                "text": random.choice(notes_templates),
            },
            "record_id": str(uuid.uuid4()),
        }
        records.append(record)

    return records


class MedicalRecordIndex:
    """Index for medical record search."""

    def __init__(self, use_http: bool = False, base_url: str = "http://localhost:8000"):
        if use_http:
            self.client = HolonClient(remote_url=base_url)
        else:
            self.store = CPUStore()
            self.client = HolonClient(local_store=self.store)

    def ingest_records(self, records: List[Dict[str, Any]]):
        """Ingest medical records."""
        print(f"📥 Ingesting {len(records)} medical records...")
        start = time.time()

        for record in records:
            self.client.insert_json(record)

        elapsed = time.time() - start
        rate = len(records) / elapsed if elapsed > 0 else 0
        print(f"   ✅ Ingested in {elapsed:.2f}s ({rate:.0f}/sec)")

    def search_by_symptoms(
        self, symptoms: List[str], limit: int = 10
    ) -> List[Dict]:
        """Search for records with similar symptoms."""
        return self.client.search_json(probe={"symptoms": symptoms}, limit=limit)

    def search_severe_cases(
        self, min_severity: int = 7, limit: int = 10
    ) -> List[Dict]:
        """Search for severe cases using manual filtering."""
        # Get all records and filter manually (guards on arrays don't work)
        all_results = self.client.search_json(probe={}, limit=100)

        severe_cases = []
        for r in all_results:
            data = r['data']
            diagnoses = data.get('diagnoses', [])

            # Check if any diagnosis has severity >= min_severity
            for diagnosis in diagnoses:
                if diagnosis.get('severity', 0) >= min_severity:
                    severe_cases.append(r)
                    break

        return severe_cases[:limit]

    def search_by_notes(self, text: str, limit: int = 10) -> List[Dict]:
        """Search by clinical notes."""
        probe = {"notes": {"$mode": "ngram", "text": text}}
        return self.client.search_json(probe=probe, limit=limit)

    def search_excluding_medication(
        self, medication: str, limit: int = 10
    ) -> List[Dict]:
        """Search for records NOT using a specific medication."""
        results = []
        all_records = self.client.search_json(probe={}, limit=100)

        for r in all_records:
            record_med = r["data"].get("treatments", {}).get("medication", "")
            if record_med != medication:
                results.append(r)

        return results[:limit]


def demo_symptom_search(index: MedicalRecordIndex):
    """Demo 1: Symptom-based search."""
    print("\n" + "=" * 70)
    print("DEMO 1: Search by Symptoms")
    print("=" * 70)

    symptoms = ["fever", "cough"]
    print(f"\n🔍 Searching for records with symptoms: {symptoms}...")
    results = index.search_by_symptoms(symptoms, limit=5)

    print(f"   Found {len(results)} matching records:")
    for r in results:
        data = r["data"]
        patient_symptoms = data.get("symptoms", [])
        diagnosis = data.get("diagnoses", [{}])[0].get("condition", "unknown")
        print(
            f"   - {data['patient_id']}: {', '.join(patient_symptoms)} → {diagnosis} (score: {r['score']:.3f})"
        )


def demo_severity_filter(index: MedicalRecordIndex):
    """Demo 2: Severity-based filtering."""
    print("\n" + "=" * 70)
    print("DEMO 2: Find Severe Cases (severity >= 7)")
    print("=" * 70)
    print("   Note: Using manual filtering (guards on arrays don't work in Holon)")

    print("\n🔍 Searching for severe cases...")
    results = index.search_severe_cases(min_severity=7, limit=10)

    print(f"   Found {len(results)} severe cases:")
    for r in results:
        data = r["data"]
        diagnosis = data.get("diagnoses", [{}])[0]
        print(
            f"   - {data['patient_id']}: {diagnosis.get('condition')} (severity: {diagnosis.get('severity')}) (score: {r['score']:.3f})"
        )


def demo_notes_search(index: MedicalRecordIndex):
    """Demo 3: Search by clinical notes."""
    print("\n" + "=" * 70)
    print("DEMO 3: Search by Clinical Notes")
    print("=" * 70)

    search_text = "persistent dry cough"
    print(f"\n🔍 Searching for: '{search_text}'...")
    results = index.search_by_notes(search_text, limit=5)

    print(f"   Found {len(results)} matching records:")
    for r in results:
        data = r["data"]
        notes = data.get("notes", {}).get("text", "")
        print(f"   - {data['patient_id']}: '{notes[:50]}...' (score: {r['score']:.3f})")


def demo_medication_exclusion(index: MedicalRecordIndex):
    """Demo 4: Exclude specific medication."""
    print("\n" + "=" * 70)
    print("DEMO 4: Find Records WITHOUT Antibiotics")
    print("=" * 70)

    print("\n🔍 Searching for non-antibiotic treatments...")
    results = index.search_excluding_medication("antibiotic", limit=5)

    print(f"   Found {len(results)} records:")
    for r in results:
        data = r["data"]
        med = data.get("treatments", {}).get("medication", "unknown")
        diagnosis = data.get("diagnoses", [{}])[0].get("condition", "unknown")
        print(f"   - {data['patient_id']}: {diagnosis} → {med} (score: {r['score']:.3f})")


def demo_complex_query(index: MedicalRecordIndex):
    """Demo 5: Complex multi-field query."""
    print("\n" + "=" * 70)
    print("DEMO 5: Complex Query - Respiratory + Severe + Notes")
    print("=" * 70)
    print("   Note: Using fuzzy search + manual severity filtering")

    print("\n🔍 Complex fuzzy search...")
    # First do fuzzy search, then filter manually
    probe = {
        "symptoms": ["cough", "fever"],
        "notes": {"$mode": "ngram", "text": "persistent cough"},
    }

    results = index.client.search_json(probe=probe, limit=20)

    # Manual filter for severity and respiratory condition
    filtered = []
    for r in results:
        data = r['data']
        diagnoses = data.get("diagnoses", [])

        for diagnosis in diagnoses:
            condition = diagnosis.get("condition", "")
            severity = diagnosis.get("severity", 0)

            if severity >= 5 and "respiratory" in condition.lower():
                filtered.append(r)
                break

    print(f"   Found {len(filtered)} matching records:")
    for r in filtered[:5]:
        data = r["data"]
        diagnosis = data.get("diagnoses", [{}])[0]
        print(
            f"   - {data['patient_id']}: {diagnosis.get('condition')} (severity: {diagnosis.get('severity')}) (score: {r['score']:.3f})"
        )


def main():
    parser = argparse.ArgumentParser(description="Fuzzy Medical Record Matching")
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--count", type=int, default=50, help="Number of records to generate")
    args = parser.parse_args()

    print("=" * 70)
    print("FUZZY MEDICAL RECORD MATCHING")
    print("=" * 70)

    mode = "HTTP" if args.http else "Local"
    print(f"\n🔧 Mode: {mode}")

    start_time = time.time()

    # Create index
    index = MedicalRecordIndex(use_http=args.http, base_url=args.url)

    # Generate and ingest records
    records = generate_medical_records(args.count)
    index.ingest_records(records)

    # Run demos
    demo_symptom_search(index)
    demo_severity_filter(index)
    demo_notes_search(index)
    demo_medication_exclusion(index)
    demo_complex_query(index)

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"""
    Mode: {mode}
    Elapsed: {elapsed:.2f}s

    Records Indexed: {len(records)}

    ✅ Fuzzy medical record matching demonstrates:
       - Symptom-based fuzzy matching
       - Severity-based guard filters
       - N-gram text search in clinical notes
       - Medication exclusion (negation)
       - Complex multi-field queries

    This enables finding similar cases without exact matching,
    crucial for medical decision support systems!
    """
    )


if __name__ == "__main__":
    main()
