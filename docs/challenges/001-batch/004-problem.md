# Spell & Magic Item Semantic Search

Goal: Store D&D 5e style spells/items and enable creative, fuzzy lookup.

Structure:
- :name
- :level (0–9 or "cantrip")
- :school (keyword)
- :classes (set of keywords)
- :casting_time
- :range
- :components (set :v :s :m)
- :duration
- :description (short text)
- :tags (set e.g. #{"damage" "control" "illusion" "concentration"})

Queries to support:
- Spells like "fireball" but concentration and lower level
- Cantrips useful for stealth / illusion
- Magic items that give flight or teleport, under legendary rarity
- Anything with "charm" in name or effect, NOT affecting humanoids

Use Holon's guards, negations, and fuzzy similarity on both structure and text.

Show full code with at least 15 spells/items ingested and 4–5 query demos.
