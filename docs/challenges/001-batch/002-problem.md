# Holon-Powered Recipe Memory & Substitution Finder

Goal: Store recipes as structured data and enable smart similarity search + ingredient substitution.

Recipe structure:
- :name        (string)
- :cuisine     (keyword e.g. :italian :mexican :asian)
- :diet        (set e.g. #{"vegan" "gluten-free"})
- :difficulty  (keyword :easy :medium :hard)
- :time        (integer minutes)
- :ingredients (vector of maps: {:item string :amount number :unit keyword})
- :tags        (set e.g. #{"quick" "comfort" "spicy"})

Example tasks:
1. Store 10–15 diverse recipes (invent them)
2. Answer queries such as:
   - Find recipes most similar to "classic lasagna" but vegan
   - Recipes similar to pad thai, but without shrimp (negation) and under 45 min
   - "What can replace tofu in this mapo tofu recipe?" → find structurally similar dishes that use different proteins
   - Wildcard: dishes with "curry" anywhere in name or tags, medium+ difficulty

Use Holon's structural encoding, fuzzy matching, guards on numeric fields (time), and negations.

Show encoding choices (how to bind ingredients, how to handle lists/vectors), ingestion, and several query examples with ranked results.
