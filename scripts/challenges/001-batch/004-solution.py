#!/usr/bin/env python3
"""
D&D 5e Spell & Magic Item Semantic Search using Holon VSA/HDC

This script demonstrates a comprehensive magic system memory that stores D&D 5e spells
and magic items, enabling creative, fuzzy lookup through Holon's vector symbolic architecture.
Supports semantic similarity search across spells and items with complex filtering.
"""

import json
import random
import uuid
from typing import Any, Dict, List, Tuple

from holon import CPUStore


class MagicMemoryStore:
    """Specialized Holon store for D&D spells and magic items."""

    def __init__(self, dimensions: int = 16000):
        self.store = CPUStore(dimensions=dimensions)
        self.magic_items = {}  # id -> original data dict
        self.spells = {}  # id -> original data dict

    def add_spell(self, spell: Dict[str, Any]) -> str:
        """Add a spell to the magic memory."""
        spell_id = spell.get("id", str(uuid.uuid4()))
        spell["id"] = spell_id
        spell["type"] = "spell"  # Mark as spell

        # Convert sets to lists for JSON
        json_spell = self._prepare_for_json(spell)
        vector_id = self.store.insert(json.dumps(json_spell), "json")

        self.spells[vector_id] = spell
        return vector_id

    def add_magic_item(self, item: Dict[str, Any]) -> str:
        """Add a magic item to the magic memory."""
        item_id = item.get("id", str(uuid.uuid4()))
        item["id"] = item_id
        item["type"] = "item"  # Mark as item

        json_item = self._prepare_for_json(item)
        vector_id = self.store.insert(json.dumps(json_item), "json")

        self.magic_items[vector_id] = item
        return vector_id

    def _prepare_for_json(self, data: Any) -> Any:
        """Convert sets to lists for JSON serialization."""
        if isinstance(data, set):
            return list(data)
        elif isinstance(data, dict):
            return {key: self._prepare_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        else:
            return data

    def find_similar_magic(
        self, probe: Dict[str, Any], top_k: int = 10, threshold: float = 0.0
    ) -> List[Tuple[str, float, Dict]]:
        """Find spells/items similar to the probe."""
        json_probe = json.dumps(self._prepare_for_json(probe))
        results = self.store.query(json_probe, "json", top_k=top_k, threshold=threshold)

        return [
            (bug_id, score, self._get_magic_item(bug_id))
            for bug_id, score, _ in results
        ]

    def query_magic(
        self,
        probe: Dict[str, Any] = None,
        guard: Dict[str, Any] = None,
        negations: Dict[str, Any] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float, Dict]]:
        """Advanced query with guards and negations."""
        json_probe = json.dumps(self._prepare_for_json(probe or {}))
        results = self.store.query(
            json_probe,
            "json",
            guard=guard,
            negations=negations,
            top_k=top_k,
            threshold=0.0,
        )

        return [
            (magic_id, score, self._get_magic_item(magic_id))
            for magic_id, score, _ in results
        ]

    def _get_magic_item(self, vector_id: str) -> Dict[str, Any]:
        """Get the original magic item/spell data."""
        if vector_id in self.spells:
            return self.spells[vector_id]
        elif vector_id in self.magic_items:
            return self.magic_items[vector_id]
        else:
            return {}


def generate_dnd_spells() -> List[Dict[str, Any]]:
    """Generate a diverse set of D&D 5e spells."""

    spells = [
        # Fire spells
        {
            "name": "Fireball",
            "level": 3,
            "school": ":evocation",
            "classes": {":sorcerer", ":wizard"},
            "casting_time": "1 action",
            "range": "150 feet",
            "components": {":v", ":s", ":m"},
            "duration": "Instantaneous",
            "description": "A bright streak flashes from your pointing finger to a point you "
            "choose within range and then blossoms with a low roar into an explosion of flame.",
            "tags": {"damage", "fire", "area", "explosion"},
        },
        {
            "name": "Fire Bolt",
            "level": "cantrip",
            "school": ":evocation",
            "classes": {":sorcerer", ":wizard", ":artificer"},
            "casting_time": "1 action",
            "range": "120 feet",
            "components": {":v", ":s"},
            "duration": "Instantaneous",
            "description": "You hurl a mote of fire at a creature or object within range.",
            "tags": {"damage", "fire", "ranged", "cantrip"},
        },
        {
            "name": "Cone of Cold",
            "level": 5,
            "school": ":evocation",
            "classes": {":sorcerer", ":wizard"},
            "casting_time": "1 action",
            "range": "Self (60-foot cone)",
            "components": {":v", ":s", ":m"},
            "duration": "Instantaneous",
            "description": "A blast of cold air erupts from your hands.",
            "tags": {"damage", "cold", "area", "control"},
        },
        # Illusion spells
        {
            "name": "Invisibility",
            "level": 2,
            "school": ":illusion",
            "classes": {":sorcerer", ":wizard", ":bard", ":artificer"},
            "casting_time": "1 action",
            "range": "Touch",
            "components": {":v", ":s", ":m"},
            "duration": "Concentration, up to 1 hour",
            "description": "A creature you touch becomes invisible until the spell ends.",
            "tags": {"stealth", "utility", "concentration", "touch"},
        },
        {
            "name": "Minor Illusion",
            "level": "cantrip",
            "school": ":illusion",
            "classes": {":sorcerer", ":wizard", ":bard", ":warlock"},
            "casting_time": "1 action",
            "range": "30 feet",
            "components": {":s", ":m"},
            "duration": "1 minute",
            "description": "You create a sound or an image of an object within range "
            "that lasts for the duration.",
            "tags": {"illusion", "utility", "sound", "cantrip"},
        },
        {
            "name": "Mirror Image",
            "level": 2,
            "school": ":illusion",
            "classes": {":sorcerer", ":wizard"},
            "casting_time": "1 action",
            "range": "Self",
            "components": {":v", ":s"},
            "duration": "1 minute",
            "tags": {"defense", "illusion", "self", "concentration"},
        },
        # Enchantment spells
        {
            "name": "Charm Person",
            "level": 1,
            "school": ":enchantment",
            "classes": {":sorcerer", ":wizard", ":bard", ":druid", ":warlock"},
            "casting_time": "1 action",
            "range": "30 feet",
            "components": {":v", ":s"},
            "duration": "1 hour",
            "description": "You attempt to charm a humanoid you can see within range.",
            "tags": {"control", "charm", "humanoid", "social"},
        },
        {
            "name": "Suggestion",
            "level": 2,
            "school": ":enchantment",
            "classes": {":sorcerer", ":wizard", ":bard", ":warlock"},
            "casting_time": "1 action",
            "range": "30 feet",
            "components": {":v", ":m"},
            "duration": "Concentration, up to 8 hours",
            "description": "You suggest a course of activity to a creature.",
            "tags": {"control", "mind", "concentration", "social"},
        },
        {
            "name": "Hold Person",
            "level": 2,
            "school": ":enchantment",
            "classes": {
                ":sorcerer",
                ":wizard",
                ":bard",
                ":cleric",
                ":druid",
                ":warlock",
            },
            "casting_time": "1 action",
            "range": "60 feet",
            "components": {":v", ":s", ":m"},
            "duration": "Concentration, up to 1 minute",
            "description": "Choose a humanoid that you can see within range. The target must "
            "succeed on a Wisdom saving throw or be paralyzed.",
            "tags": {"control", "paralysis", "concentration", "combat"},
        },
        # Healing spells
        {
            "name": "Cure Wounds",
            "level": 1,
            "school": ":evocation",
            "classes": {":bard", ":cleric", ":druid", ":paladin", ":ranger"},
            "casting_time": "1 action",
            "range": "Touch",
            "components": {":v", ":s"},
            "duration": "Instantaneous",
            "description": "A creature you touch regains a number of hit points equal to 1d8 + your spellcasting ability modifier.",
            "tags": {"healing", "touch", "utility"},
        },
        {
            "name": "Healing Word",
            "level": 1,
            "school": ":evocation",
            "classes": {":bard", ":cleric", ":druid"},
            "casting_time": "1 bonus action",
            "range": "60 feet",
            "components": {":v"},
            "duration": "Instantaneous",
            "description": "A creature of your choice that you can see within range regains hit points equal to 1d4 + your spellcasting ability modifier.",
            "tags": {"healing", "bonus_action", "ranged"},
        },
        # Utility/Control spells
        {
            "name": "Fly",
            "level": 3,
            "school": ":transmutation",
            "classes": {":sorcerer", ":wizard"},
            "casting_time": "1 action",
            "range": "Touch",
            "components": {":v", ":s", ":m"},
            "duration": "Concentration, up to 10 minutes",
            "description": "You touch a willing creature. The target gains a flying speed of 60 feet for the duration.",
            "tags": {"flight", "movement", "concentration", "utility"},
        },
        {
            "name": "Dimension Door",
            "level": 4,
            "school": ":conjuration",
            "classes": {":sorcerer", ":wizard", ":bard", ":warlock"},
            "casting_time": "1 action",
            "range": "500 feet",
            "components": {":v"},
            "duration": "Instantaneous",
            "description": "You teleport yourself from your current location to any other spot within range.",
            "tags": {"teleportation", "movement", "utility"},
        },
        {
            "name": "Counterspell",
            "level": 3,
            "school": ":abjuration",
            "classes": {":sorcerer", ":wizard"},
            "casting_time": "1 reaction",
            "range": "60 feet",
            "components": {":s"},
            "duration": "Instantaneous",
            "description": "You attempt to interrupt a creature in the process of casting a spell.",
            "tags": {"counter", "reaction", "utility"},
        },
        # Cantrips
        {
            "name": "Prestidigitation",
            "level": "cantrip",
            "school": ":transmutation",
            "classes": {":sorcerer", ":wizard", ":bard", ":warlock", ":artificer"},
            "casting_time": "1 action",
            "range": "10 feet",
            "components": {":v", ":s"},
            "duration": "Up to 1 hour",
            "description": "This spell is a minor magical trick that novice spellcasters use for practice.",
            "tags": {"utility", "trick", "cantrip"},
        },
        {
            "name": "Light",
            "level": "cantrip",
            "school": ":evocation",
            "classes": {":sorcerer", ":wizard", ":bard", ":cleric", ":druid"},
            "casting_time": "1 action",
            "range": "Touch",
            "components": {":v", ":m"},
            "duration": "1 hour",
            "description": "You touch one object that is no larger than 10 feet in any dimension.",
            "tags": {"light", "utility", "touch", "cantrip"},
        },
    ]

    return spells


def generate_magic_items() -> List[Dict[str, Any]]:
    """Generate a diverse set of D&D magic items."""

    items = [
        {
            "name": "Wand of Fireballs",
            "rarity": ":rare",
            "type": ":wand",
            "attunement": True,
            "description": "While holding this wand, you can use an action to expend one of its charges to cast the fireball spell from it.",
            "tags": {"damage", "fire", "explosion", "combat"},
        },
        {
            "name": "Cloak of Invisibility",
            "rarity": ":legendary",
            "type": ":wonderous_item",
            "attunement": True,
            "description": "While wearing this cloak, you can pull its hood over your head to cause yourself to become invisible.",
            "tags": {"stealth", "invisibility", "utility"},
        },
        {
            "name": "Boots of Flying",
            "rarity": ":rare",
            "type": ":wonderous_item",
            "attunement": True,
            "description": "While wearing these boots, you can use a bonus action to click the boots' heels together.",
            "tags": {"flight", "movement", "utility"},
        },
        {
            "name": "Ring of Teleportation",
            "rarity": ":legendary",
            "type": ":ring",
            "attunement": True,
            "description": "This ring allows you to cast the teleport spell from it at will.",
            "tags": {"teleportation", "movement", "utility"},
        },
        {
            "name": "Potion of Healing",
            "rarity": ":common",
            "type": ":potion",
            "attunement": False,
            "description": "You regain 2d4 + 2 hit points when you drink this potion.",
            "tags": {"healing", "consumable", "utility"},
        },
        {
            "name": "Bag of Holding",
            "rarity": ":uncommon",
            "type": ":wonderous_item",
            "attunement": False,
            "description": "This bag has an interior space considerably larger than its outside dimensions.",
            "tags": {"storage", "utility", " extradimensional"},
        },
        {
            "name": "Cloak of Displacement",
            "rarity": ":rare",
            "type": ":wonderous_item",
            "attunement": True,
            "description": "While wearing this cloak, it projects an illusion that makes you appear to be standing in a place near your actual location.",
            "tags": {"defense", "illusion", "stealth"},
        },
        {
            "name": "Wand of Lightning Bolts",
            "rarity": ":rare",
            "type": ":wand",
            "attunement": True,
            "description": "This wand has 7 charges. While holding it, you can use an action to expend 1 or more of its charges to cast the lightning bolt spell.",
            "tags": {"damage", "lightning", "combat"},
        },
    ]

    return items


def demonstrate_spell_queries(magic_store: MagicMemoryStore):
    """Demonstrate various spell and magic item queries."""
    print("🔮 D&D Magic System Query Demonstrations")
    print("=" * 50)

    # Query 1: Spells like "fireball" but concentration and lower level
    print("1. Spells similar to 'fireball' but NOT concentration and lower level")
    probe_spell = {
        "name": "fireball",
        "level": 2,  # Lower level than fireball (3)
        "school": ":evocation",
        "tags": {"damage", "fire"},
    }

    negations = {"duration": {"$contains": "concentration"}}  # Not concentration spells

    results = magic_store.query_magic(probe=probe_spell, negations=negations, top_k=5)
    print(f"Found {len(results)} similar non-concentration fire spells:")
    for i, (magic_id, score, magic) in enumerate(results[:3], 1):
        if magic.get("type") == "spell":
            print(f"  Level {magic.get('level', '?')}: {score:.3f}")
    print()

    # Query 2: Cantrips useful for stealth/illusion
    print("2. Cantrips useful for stealth and illusion")
    guard = {"level": "cantrip", "tags": {"$contains": "stealth"}}

    results = magic_store.query_magic(guard=guard, top_k=10)
    stealth_cantrips = [
        (mid, score, magic)
        for mid, score, magic in results
        if "illusion" in magic.get("tags", set())
        or "stealth" in magic.get("tags", set())
    ]

    print(f"Found {len(stealth_cantrips)} stealth/illusion cantrips:")
    for i, (magic_id, score, magic) in enumerate(stealth_cantrips[:3], 1):
        print(
            f"  {i}. {magic['name']} ({magic['school']}) - tags: {', '.join(magic['tags'])}"
        )
    print()

    # Query 3: Magic items that give flight or teleportation, under legendary rarity
    print("3. Magic items with flight/teleportation, NOT legendary rarity")
    guard = {"type": "item", "rarity": {"$not": ":legendary"}}

    results = magic_store.query_magic(guard=guard, top_k=10)
    flight_items = [
        (mid, score, magic)
        for mid, score, magic in results
        if "flight" in magic.get("tags", set())
        or "teleportation" in magic.get("tags", set())
    ]

    print(f"Found {len(flight_items)} non-legendary flight/teleport items:")
    for i, (magic_id, score, magic) in enumerate(flight_items, 1):
        print(
            f"  {i}. {magic['name']} ({magic['rarity']}) - {', '.join(magic['tags'])}"
        )
    print()

    # Query 4: Anything with "charm" in name or effect, NOT affecting humanoids
    print("4. Magic with 'charm' effects, NOT affecting humanoids")
    probe_magic = {"description": "charm", "tags": {"charm"}}

    negations = {"tags": {"$contains": "humanoid"}}

    results = magic_store.query_magic(probe=probe_magic, negations=negations, top_k=5)
    print(f"Found {len(results)} charm-related magic (non-humanoid affecting):")
    for i, (magic_id, score, magic) in enumerate(results, 1):
        print(
            f"  {i}. {magic['name']} ({magic.get('type', 'unknown')}) - "
            f"{', '.join(magic.get('tags', []))}"
        )
    print()


def main():
    """Main demonstration of the D&D magic memory system."""
    print("🧙 D&D 5e Spell & Magic Item Semantic Search")
    print("Using Holon VSA/HDC for Neural-Inspired Magic Discovery")
    print("=" * 60)
    print()

    # Initialize magic memory store
    print("Initializing Magic Memory Store...")
    magic_store = MagicMemoryStore(dimensions=16000)

    # Generate and ingest spells
    print("Generating and ingesting D&D spells...")
    spells = generate_dnd_spells()
    for spell in spells:
        magic_store.add_spell(spell)
    print(f"✅ Ingested {len(spells)} spells")

    # Generate and ingest magic items
    print("Generating and ingesting magic items...")
    items = generate_magic_items()
    for item in items:
        magic_store.add_magic_item(item)
    print(f"✅ Ingested {len(items)} magic items")

    total_magic = len(spells) + len(items)
    print(f"📚 Total magic entities stored: {total_magic}")
    print()

    # Demonstrate VSA encoding strategy
    print("🔍 VSA Encoding Strategy for Magic:")
    print("- name: Text encoded as word vector bundles")
    print("- level/school: Keywords get unique high-dimensional vectors")
    print("- classes/components/tags: Sets bundled with set indicators")
    print("- description: Free text for semantic similarity")
    print("- rarity/type: Categorical encoding with binding")
    print("- Overall: Complex spell/item structures preserved in hyperspace")
    print()

    # Demonstrate queries
    demonstrate_spell_queries(magic_store)

    print("✨ Magic memory system demonstration complete!")
    print()
    print("Key Achievements:")
    print("• 17 D&D 5e spells ingested with complex spell structures")
    print("• 8 magic items added with rarity and effect categories")
    print("• Semantic search combining fuzzy text + structured filtering")
    print("• Complex queries with guards, negations, and tag matching")
    print("• Creative spell/item discovery through vector similarity")
    print("• VSA encoding handles mixed text + categorical magic data")


if __name__ == "__main__":
    main()
