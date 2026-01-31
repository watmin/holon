#!/usr/bin/env python3
"""
D&D 5e Spell & Magic Item Semantic Search - HTTP API Version

This demonstrates the magic item search system working via Holon's HTTP API,
proving the solution works with both in-memory and remote deployments.
"""

import json

import requests

BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"


class HTTPMagicStore:
    """HTTP client for magic items that works with Holon server."""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.api_prefix = API_PREFIX
        self.spells = {}
        self.items = {}

    def health_check(self) -> bool:
        """Check if server is running."""
        try:
            response = requests.get(
                f"{self.base_url}{self.api_prefix}/health", timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def insert_spell(self, spell: dict) -> str:
        """Insert a spell via HTTP API."""
        data = self._prepare_for_json(spell)
        response = requests.post(
            f"{self.base_url}{self.api_prefix}/items",
            json={"data": json.dumps(data), "data_type": "json"},
            timeout=10,
        )
        response.raise_for_status()
        spell_id = response.json()["id"]
        self.spells[spell_id] = spell
        return spell_id

    def insert_item(self, item: dict) -> str:
        """Insert a magic item via HTTP API."""
        data = self._prepare_for_json(item)
        response = requests.post(
            f"{self.base_url}{self.api_prefix}/items",
            json={"data": json.dumps(data), "data_type": "json"},
            timeout=10,
        )
        response.raise_for_status()
        item_id = response.json()["id"]
        self.items[item_id] = item
        return item_id

    def search(
        self, query: dict, limit: int = 10, threshold: float = 0.0
    ) -> list:
        """Search magic items via HTTP API."""
        payload = {
            "probe": json.dumps(self._prepare_for_json(query)),
            "data_type": "json",
            "top_k": limit,
            "threshold": threshold,
        }

        response = requests.post(
            f"{self.base_url}{self.api_prefix}/search",
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        results = response.json()["results"]

        # Enrich with cached data
        enriched = []
        for r in results:
            data = self.spells.get(r["id"]) or self.items.get(r["id"]) or r.get("data", {})
            enriched.append({"id": r["id"], "score": r["score"], "data": data})
        return enriched

    def _prepare_for_json(self, data):
        """Convert sets to lists for JSON serialization."""
        if isinstance(data, set):
            return list(data)
        elif isinstance(data, dict):
            return {k: self._prepare_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        return data


def create_sample_spells():
    """Generate sample D&D spells."""
    return [
        {
            "name": "Fireball",
            "level": 3,
            "school": ":evocation",
            "classes": {"wizard", "sorcerer"},
            "casting_time": "1 action",
            "range": "150 feet",
            "components": {"v", "s", "m"},
            "duration": "Instantaneous",
            "description": "A bright streak flashes creating a massive explosion of flame",
            "tags": {"damage", "fire", "aoe"},
        },
        {
            "name": "Shield",
            "level": 1,
            "school": ":abjuration",
            "classes": {"wizard", "sorcerer"},
            "casting_time": "1 reaction",
            "range": "Self",
            "components": {"v", "s"},
            "duration": "1 round",
            "description": "An invisible barrier of magical force protects you",
            "tags": {"defense", "reaction"},
        },
        {
            "name": "Minor Illusion",
            "level": 0,
            "school": ":illusion",
            "classes": {"wizard", "sorcerer", "bard", "warlock"},
            "casting_time": "1 action",
            "range": "30 feet",
            "components": {"s", "m"},
            "duration": "1 minute",
            "description": "Create a sound or image of an object",
            "tags": {"illusion", "utility", "cantrip"},
        },
        {
            "name": "Healing Word",
            "level": 1,
            "school": ":evocation",
            "classes": {"bard", "cleric", "druid"},
            "casting_time": "1 bonus action",
            "range": "60 feet",
            "components": {"v"},
            "duration": "Instantaneous",
            "description": "Speak a word of healing to restore hit points",
            "tags": {"healing", "bonus action"},
        },
        {
            "name": "Counterspell",
            "level": 3,
            "school": ":abjuration",
            "classes": {"wizard", "sorcerer", "warlock"},
            "casting_time": "1 reaction",
            "range": "60 feet",
            "components": {"s"},
            "duration": "Instantaneous",
            "description": "Interrupt a creature casting a spell",
            "tags": {"reaction", "counter"},
        },
    ]


def create_sample_items():
    """Generate sample magic items."""
    return [
        {
            "name": "Boots of Flying",
            "type": ":wondrous",
            "rarity": ":rare",
            "attunement": True,
            "description": "While wearing these boots, you can fly",
            "tags": {"flight", "movement", "utility"},
        },
        {
            "name": "Bag of Holding",
            "type": ":wondrous",
            "rarity": ":uncommon",
            "attunement": False,
            "description": "This bag has an interior space larger than outside",
            "tags": {"utility", "storage", "extradimensional"},
        },
        {
            "name": "Wand of Fireballs",
            "type": ":wand",
            "rarity": ":rare",
            "attunement": True,
            "description": "This wand has 7 charges for casting fireball",
            "tags": {"fire", "damage", "combat"},
        },
    ]


def main():
    """Main demonstration."""
    print("🧙 D&D Magic Search - HTTP API Demo")
    print("=" * 60)

    store = HTTPMagicStore()

    # Health check
    print("\n🔗 Checking Holon HTTP service...")
    if not store.health_check():
        print("❌ Server not running. Start with:")
        print("   ./scripts/run_with_venv.sh python scripts/server/holon_server.py")
        return

    health = requests.get(f"{BASE_URL}{API_PREFIX}/health").json()
    print(f"✅ Connected: {health['status']} | Backend: {health['backend']}")

    # Insert spells
    spells = create_sample_spells()
    print(f"\n📥 Inserting {len(spells)} spells via HTTP...")
    for spell in spells:
        store.insert_spell(spell)
    print(f"✅ Inserted {len(spells)} spells")

    # Insert items
    items = create_sample_items()
    print(f"📥 Inserting {len(items)} magic items via HTTP...")
    for item in items:
        store.insert_item(item)
    print(f"✅ Inserted {len(items)} items")

    # Query demonstrations
    print("\n" + "=" * 60)
    print("🔍 QUERY DEMONSTRATIONS VIA HTTP")
    print("=" * 60)

    # 1. Similar to fireball
    print("\n1. SPELLS LIKE FIREBALL: Fire damage spells")
    results = store.search({"name": "fireball", "tags": ["fire", "damage"]}, limit=3)
    for r in results:
        data = r["data"]
        print(f"   [{r['score']:.3f}] {data.get('name', 'N/A')} (Level {data.get('level', '?')})")

    # 2. Illusion spells
    print("\n2. ILLUSION MAGIC: Illusion school spells")
    results = store.search({"school": ":illusion"}, limit=3)
    for r in results:
        data = r["data"]
        print(f"   [{r['score']:.3f}] {data.get('name', 'N/A')}")

    # 3. Reaction spells
    print("\n3. REACTION SPELLS: Quick response spells")
    results = store.search({"tags": ["reaction"]}, limit=3)
    for r in results:
        data = r["data"]
        print(f"   [{r['score']:.3f}] {data.get('name', 'N/A')}")

    # 4. Utility items
    print("\n4. UTILITY ITEMS: Useful magic items")
    results = store.search({"tags": ["utility"]}, limit=3)
    for r in results:
        data = r["data"]
        print(f"   [{r['score']:.3f}] {data.get('name', 'N/A')}")

    # 5. Flight-related
    print("\n5. FLIGHT MAGIC: Flying capabilities")
    results = store.search({"tags": ["flight"]}, limit=3)
    for r in results:
        data = r["data"]
        print(f"   [{r['score']:.3f}] {data.get('name', 'N/A')}")

    print("\n" + "=" * 60)
    print("✅ HTTP API DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey achievements:")
    print("   - Spells and items stored via HTTP API")
    print("   - Semantic search finds similar magic")
    print("   - Same queries work in-memory OR via HTTP")
    print("   - Ready for production deployment")


if __name__ == "__main__":
    main()
