#!/usr/bin/env python3
"""
Recipe Memory & Substitution Finder using Holon VSA/HDC

This script demonstrates a smart recipe memory system that can store recipes
as structured data and enable similarity search, ingredient substitution,
and advanced querying using Holon's vector symbolic architecture.
"""

import json

from holon import CPUStore, HolonClient


def create_sample_recipes():
    """Generate 12 diverse recipes with varied cuisines, diets, and ingredients."""

    recipes = [
        # Italian recipes
        {
            "name": "Classic Lasagna",
            "cuisine": ":italian",
            "diet": set(),  # empty set means no special diet
            "difficulty": ":medium",
            "time": 90,
            "ingredients": [
                {"item": "ground beef", "amount": 1.0, "unit": ":lb"},
                {"item": "lasagna noodles", "amount": 12, "unit": ":sheets"},
                {"item": "ricotta cheese", "amount": 15, "unit": ":oz"},
                {"item": "mozzarella cheese", "amount": 8, "unit": ":oz"},
                {"item": "parmesan cheese", "amount": 0.5, "unit": ":cup"},
                {"item": "tomato sauce", "amount": 24, "unit": ":oz"},
                {"item": "onion", "amount": 1, "unit": ":medium"},
                {"item": "garlic", "amount": 3, "unit": ":cloves"},
                {"item": "italian seasoning", "amount": 1, "unit": ":tbsp"},
            ],
            "tags": {"comfort", "family", "baking"},
        },
        {
            "name": "Vegan Eggplant Parmesan",
            "cuisine": ":italian",
            "diet": {"vegan"},
            "difficulty": ":medium",
            "time": 60,
            "ingredients": [
                {"item": "eggplant", "amount": 2, "unit": ":large"},
                {"item": "vegan mozzarella", "amount": 8, "unit": ":oz"},
                {"item": "bread crumbs", "amount": 1, "unit": ":cup"},
                {"item": "tomato sauce", "amount": 24, "unit": ":oz"},
                {"item": "basil", "amount": 0.25, "unit": ":cup"},
                {"item": "olive oil", "amount": 3, "unit": ":tbsp"},
                {"item": "garlic powder", "amount": 1, "unit": ":tsp"},
            ],
            "tags": {"vegan", "italian", "vegetarian"},
        },
        # Asian recipes
        {
            "name": "Pad Thai",
            "cuisine": ":asian",
            "diet": "#{}",  # can be made vegan
            "difficulty": ":medium",
            "time": 30,
            "ingredients": [
                {"item": "rice noodles", "amount": 8, "unit": ":oz"},
                {"item": "shrimp", "amount": 0.5, "unit": ":lb"},
                {"item": "tofu", "amount": 8, "unit": ":oz"},
                {"item": "bean sprouts", "amount": 2, "unit": ":cups"},
                {"item": "peanuts", "amount": 0.25, "unit": ":cup"},
                {"item": "lime", "amount": 1, "unit": ":whole"},
                {"item": "fish sauce", "amount": 2, "unit": ":tbsp"},
                {"item": "tamarind paste", "amount": 1, "unit": ":tbsp"},
                {"item": "palm sugar", "amount": 1, "unit": ":tbsp"},
                {"item": "chili flakes", "amount": 0.5, "unit": ":tsp"},
            ],
            "tags": {"thai", "noodles", "quick", "spicy"},
        },
        {
            "name": "Mapo Tofu",
            "cuisine": ":asian",
            "diet": {"vegan"},
            "difficulty": ":easy",
            "time": 25,
            "ingredients": [
                {"item": "tofu", "amount": 14, "unit": ":oz"},
                {"item": "ground pork", "amount": 0.25, "unit": ":lb"},
                {"item": "fermented bean paste", "amount": 1, "unit": ":tbsp"},
                {"item": "sichuan peppercorns", "amount": 1, "unit": ":tsp"},
                {"item": "garlic", "amount": 3, "unit": ":cloves"},
                {"item": "ginger", "amount": 1, "unit": ":tbsp"},
                {"item": "green onions", "amount": 3, "unit": ":stalks"},
                {"item": "cornstarch", "amount": 1, "unit": ":tbsp"},
                {"item": "sesame oil", "amount": 1, "unit": ":tsp"},
            ],
            "tags": {"chinese", "spicy", "quick", "comfort"},
        },
        # Mexican recipes
        {
            "name": "Chicken Tacos",
            "cuisine": ":mexican",
            "diet": {"gluten-free"},
            "difficulty": ":easy",
            "time": 20,
            "ingredients": [
                {"item": "chicken breast", "amount": 1, "unit": ":lb"},
                {"item": "corn tortillas", "amount": 8, "unit": ":pieces"},
                {"item": "avocado", "amount": 2, "unit": ":whole"},
                {"item": "lime", "amount": 2, "unit": ":whole"},
                {"item": "cilantro", "amount": 0.25, "unit": ":cup"},
                {"item": "onion", "amount": 1, "unit": ":small"},
                {"item": "cumin", "amount": 1, "unit": ":tsp"},
                {"item": "chili powder", "amount": 1, "unit": ":tsp"},
                {"item": "garlic powder", "amount": 1, "unit": ":tsp"},
            ],
            "tags": {"mexican", "quick", "gluten-free", "protein"},
        },
        {
            "name": "Vegan Burrito Bowl",
            "cuisine": ":mexican",
            "diet": {"vegan", "gluten-free"},
            "difficulty": ":easy",
            "time": 35,
            "ingredients": [
                {"item": "brown rice", "amount": 1, "unit": ":cup"},
                {"item": "black beans", "amount": 15, "unit": ":oz"},
                {"item": "corn", "amount": 1, "unit": ":cup"},
                {"item": "avocado", "amount": 1, "unit": ":whole"},
                {"item": "salsa", "amount": 0.5, "unit": ":cup"},
                {"item": "lime", "amount": 1, "unit": ":whole"},
                {"item": "cumin", "amount": 1, "unit": ":tsp"},
                {"item": "chili powder", "amount": 1, "unit": ":tsp"},
            ],
            "tags": {"mexican", "vegan", "gluten-free", "healthy"},
        },
        # Middle Eastern recipes
        {
            "name": "Chicken Shawarma",
            "cuisine": ":middle-eastern",
            "diet": {"gluten-free"},
            "difficulty": ":medium",
            "time": 45,
            "ingredients": [
                {"item": "chicken thighs", "amount": 1.5, "unit": ":lb"},
                {"item": "yogurt", "amount": 0.5, "unit": ":cup"},
                {"item": "lemon", "amount": 1, "unit": ":whole"},
                {"item": "garlic", "amount": 4, "unit": ":cloves"},
                {"item": "cumin", "amount": 1, "unit": ":tbsp"},
                {"item": "paprika", "amount": 1, "unit": ":tbsp"},
                {"item": "turmeric", "amount": 1, "unit": ":tsp"},
                {"item": "cinnamon", "amount": 0.5, "unit": ":tsp"},
                {"item": "olive oil", "amount": 2, "unit": ":tbsp"},
            ],
            "tags": {"middle-eastern", "grilled", "spicy", "protein"},
        },
        {
            "name": "Falafel Bowls",
            "cuisine": ":middle-eastern",
            "diet": {"vegan", "gluten-free"},
            "difficulty": ":medium",
            "time": 40,
            "ingredients": [
                {"item": "chickpeas", "amount": 15, "unit": ":oz"},
                {"item": "onion", "amount": 1, "unit": ":medium"},
                {"item": "garlic", "amount": 3, "unit": ":cloves"},
                {"item": "parsley", "amount": 0.5, "unit": ":cup"},
                {"item": "cumin", "amount": 1, "unit": ":tsp"},
                {"item": "coriander", "amount": 1, "unit": ":tsp"},
                {"item": "flour", "amount": 2, "unit": ":tbsp"},
                {"item": "tahini", "amount": 0.25, "unit": ":cup"},
                {"item": "lemon", "amount": 0.5, "unit": ":whole"},
            ],
            "tags": {"middle-eastern", "vegan", "fried", "healthy"},
        },
        # Indian recipes
        {
            "name": "Butter Chicken",
            "cuisine": ":indian",
            "diet": {"gluten-free"},
            "difficulty": ":medium",
            "time": 50,
            "ingredients": [
                {"item": "chicken breast", "amount": 1, "unit": ":lb"},
                {"item": "butter", "amount": 4, "unit": ":tbsp"},
                {"item": "tomato sauce", "amount": 15, "unit": ":oz"},
                {"item": "heavy cream", "amount": 0.5, "unit": ":cup"},
                {"item": "garam masala", "amount": 1, "unit": ":tbsp"},
                {"item": "cumin", "amount": 1, "unit": ":tsp"},
                {"item": "ginger", "amount": 1, "unit": ":tbsp"},
                {"item": "garlic", "amount": 3, "unit": ":cloves"},
                {"item": "onion", "amount": 1, "unit": ":medium"},
            ],
            "tags": {"indian", "curry", "creamy", "comfort"},
        },
        {
            "name": "Chana Masala",
            "cuisine": ":indian",
            "diet": {"vegan", "gluten-free"},
            "difficulty": ":easy",
            "time": 35,
            "ingredients": [
                {"item": "chickpeas", "amount": 15, "unit": ":oz"},
                {"item": "tomatoes", "amount": 2, "unit": ":medium"},
                {"item": "onion", "amount": 1, "unit": ":medium"},
                {"item": "garlic", "amount": 3, "unit": ":cloves"},
                {"item": "ginger", "amount": 1, "unit": ":tbsp"},
                {"item": "cumin", "amount": 1, "unit": ":tsp"},
                {"item": "coriander", "amount": 1, "unit": ":tsp"},
                {"item": "turmeric", "amount": 0.5, "unit": ":tsp"},
                {"item": "garam masala", "amount": 1, "unit": ":tsp"},
                {"item": "coconut oil", "amount": 1, "unit": ":tbsp"},
            ],
            "tags": {"indian", "curry", "vegan", "protein"},
        },
        # American recipes
        {
            "name": "Grilled Cheese Sandwich",
            "cuisine": ":american",
            "diet": "#{}",  # can be made vegan
            "difficulty": ":easy",
            "time": 10,
            "ingredients": [
                {"item": "bread", "amount": 2, "unit": ":slices"},
                {"item": "cheddar cheese", "amount": 2, "unit": ":slices"},
                {"item": "butter", "amount": 1, "unit": ":tbsp"},
            ],
            "tags": {"american", "quick", "comfort", "sandwich"},
        },
        {
            "name": "Vegan Mac and Cheese",
            "cuisine": ":american",
            "diet": {"vegan"},
            "difficulty": ":easy",
            "time": 25,
            "ingredients": [
                {"item": "macaroni pasta", "amount": 8, "unit": ":oz"},
                {"item": "cashews", "amount": 1, "unit": ":cup"},
                {"item": "nutritional yeast", "amount": 0.25, "unit": ":cup"},
                {"item": "lemon juice", "amount": 2, "unit": ":tbsp"},
                {"item": "garlic powder", "amount": 1, "unit": ":tsp"},
                {"item": "turmeric", "amount": 0.5, "unit": ":tsp"},
                {"item": "salt", "amount": 1, "unit": ":tsp"},
            ],
            "tags": {"american", "vegan", "comfort", "pasta"},
        },
    ]

    return recipes


def convert_recipe_to_edn(recipe):
    """Convert a Python dict recipe to EDN format string."""

    def format_value(value):
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            return str(value)
        elif isinstance(value, list):
            return f"[{', '.join(format_ingredient(ing) for ing in value)}]"
        elif isinstance(value, set):
            if value:  # Non-empty set
                return f'#{{{", ".join(f"{item}" for item in value)}}}'
            else:  # Empty set
                return '#{}'
        else:
            return str(value)

    def format_ingredient(ing):
        return f"""{{:item "{ing['item']}", :amount {ing['amount']}, :unit {ing['unit']}}}"""

    edn_parts = []
    for key, value in recipe.items():
        if key == "name":
            edn_parts.append(f':{key} "{value}"')
        elif key == "cuisine":
            edn_parts.append(f":{key} {value}")
        elif key == "diet":
            edn_parts.append(f":{key} {format_value(value)}")
        elif key == "difficulty":
            edn_parts.append(f":{key} {value}")
        elif key == "time":
            edn_parts.append(f":{key} {value}")
        elif key == "ingredients":
            edn_parts.append(
                f':{key} [{", ".join(format_ingredient(ing) for ing in value)}]'
            )
        elif key == "tags":
            edn_parts.append(f":{key} {format_value(value)}")

    return f"{{{', '.join(edn_parts)}}}"


def ingest_recipes(client, recipes):
    """Ingest recipes into the Holon store using batch operations."""
    print(f"📥 Ingesting {len(recipes)} recipes into Holon memory...")

    # Prepare all recipes for JSON serialization
    json_ready_recipes = []
    for recipe in recipes:
        recipe_dict = recipe.copy()
        # Convert sets to lists for JSON compatibility
        if 'diet' in recipe_dict and isinstance(recipe_dict['diet'], set):
            recipe_dict['diet'] = list(recipe_dict['diet'])
        if 'tags' in recipe_dict and isinstance(recipe_dict['tags'], set):
            recipe_dict['tags'] = list(recipe_dict['tags'])
        json_ready_recipes.append(recipe_dict)

    # Use batch insert for much better performance
    ids = client.insert_batch_json(json_ready_recipes)
    print(f"  ✓ Batch inserted {len(recipes)} recipes in one operation")

    print("✅ All recipes ingested successfully!")


def query_recipes(
    client, query, description, limit=10, guard=None, negations=None
):
    """Query recipes and display results."""
    print(f"\n🔍 {description}")
    print(f"Query: {query}")
    if guard:
        print(f"Guard: {guard}")
    if negations:
        print(f"Negations: {negations}")

    try:
        # Convert query string to dict if needed
        if isinstance(query, str):
            # Simple conversion for basic queries - in practice this would be more sophisticated
            query_dict = {"name": query.replace('"', '')} if '"name"' in query else {}
        else:
            query_dict = query

        results = client.search_json(
            query_dict,
            guard=guard,
            negations=negations,
            limit=limit,
            threshold=0.0,
        )

        if not results:
            print("  ❌ No matching recipes found")
            return

        print(
            f"  ✅ Found {len(results)} matching recipes (showing top {min(limit, len(results))}):"
        )

        for i, result in enumerate(results):
            recipe = result["data"]
            score = result["score"]

            print(f"  {i+1}. [{score:.3f}] {recipe['name']}")
            print(
                f"     Cuisine: {recipe['cuisine']} | Difficulty: {recipe['difficulty']} | "
                f"Time: {recipe['time']} min"
            )
            if recipe.get('diet') and recipe['diet']:
                print(f"     Diet: {recipe['diet']}")
            if recipe.get('tags') and recipe['tags']:
                print(f"     Tags: {recipe['tags']}")

    except Exception as e:
        print(f"  ❌ Query failed: {e}")


def main():
    """Main demonstration function."""
    print("🍳 Recipe Memory & Substitution Finder Demo")
    print("=" * 55)

    # Initialize Holon store and client
    print("🚀 Initializing Holon CPUStore and Client...")
    store = CPUStore(dimensions=16000)
    client = HolonClient(local_store=store)
    print("✅ Store and client initialized with 16,000 dimensions")

    # Create and ingest sample recipes
    recipes = create_sample_recipes()
    ingest_recipes(client, recipes)

    # Demonstrate various query types
    print("\n" + "=" * 55)
    print("🧪 QUERY DEMONSTRATIONS")
    print("=" * 55)

    # 1. Find recipes similar to "classic lasagna"
    query_recipes(
        client,
        {"name": "classic lasagna", "cuisine": ":italian", "difficulty": ":medium"},
        "1. FUZZY SIMILARITY: Recipes similar to classic lasagna",
        limit=5
    )

    # 2. Recipes similar to pad thai, but without shrimp
    query_recipes(
        client,
        {"name": "pad thai", "cuisine": ":asian", "difficulty": ":medium"},
        "2. SIMILARITY + NEGATION: Pad thai similar recipes, no shrimp",
        negations={"ingredients": [{"item": "shrimp"}]},
        limit=5
    )

    # 3. What can replace tofu in mapo tofu recipe? (find structurally similar dishes
    # with different proteins)
    query_recipes(
        client,
        {"name": "mapo tofu", "cuisine": ":asian", "difficulty": ":easy"},
        "3. SUBSTITUTION: Structurally similar to mapo tofu but with different main protein",
        negations={"ingredients": [{"item": "tofu"}]},
        limit=3
    )

    # 4. Dishes with "curry" in tags
    query_recipes(
        client, {"tags": ["curry"], "cuisine": ":indian"},
        "4. TAG SIMILARITY: Indian curry dishes",
        limit=3
    )

    # 5. Asian cuisine recipes
    query_recipes(client, {"cuisine": ":asian"}, "5. CUISINE FILTER: Asian recipes")

    # 6. Vegan recipes
    query_recipes(client, {"diet": ["vegan"]}, "6. DIET FILTER: Vegan recipes")

    # 7. Comfort food recipes
    query_recipes(
        client, {"tags": ["comfort"]}, "7. TAG SIMILARITY: Comfort food recipes"
    )

    # 8. Advanced OR logic: Easy recipes OR vegan recipes
    query_recipes(
        client,
        {},
        "8. ADVANCED OR LOGIC: Easy recipes OR vegan recipes",
        guard={
            "$or": [
                {"difficulty": ":easy"},
                {"diet": ["vegan"]}
            ]
        },
        limit=8
    )

    # 9. Complex guards: Easy difficulty recipes that are NOT spicy
    query_recipes(
        client,
        {},
        "9. COMPLEX GUARDS + NEGATION: Easy recipes NOT spicy",
        guard={"difficulty": ":easy"},
        negations={"tags": {"$not_contains": "spicy"}},
        limit=5
    )

    # 10. Multi-criteria similarity: Asian or Indian curry dishes
    query_recipes(
        client,
        {"tags": ["curry"]},
        "10. MULTI-CRITERIA: Asian/Indian curry dishes",
        guard={
            "$or": [
                {"cuisine": ":asian"},
                {"cuisine": ":indian"}
            ]
        },
        limit=5
    )

    # ========================================
    # NEW: Demonstrate new VSA primitives
    # ========================================
    print("\n" + "=" * 55)
    print("🆕 NEW VSA PRIMITIVE DEMONSTRATIONS")
    print("=" * 55)

    demonstrate_new_primitives(store, client, recipes)

    print("\n" + "=" * 55)
    print("🎉 Recipe Memory Demo Complete!")
    print(
        "Holon demonstrated recipe similarity, substitution, advanced OR logic, "
        "complex querying, AND new VSA primitives (prototype, difference, blend)"
    )
    print("=" * 55)


def demonstrate_new_primitives(store, client, recipes):
    """Demonstrate the new VSA primitives for recipe operations."""
    import numpy as np
    from holon.similarity import normalized_dot_similarity as cosine_similarity

    print("\n🧬 1. PROTOTYPE: Extract cuisine style patterns")
    print("-" * 55)

    # Group recipes by cuisine
    cuisines = {}
    for recipe in recipes:
        cuisine = recipe.get("cuisine", "unknown")
        if cuisine not in cuisines:
            cuisines[cuisine] = []
        cuisines[cuisine].append(store.encoder.encode_data(recipe))

    # Create prototypes for each cuisine
    cuisine_protos = {}
    for cuisine, vecs in cuisines.items():
        if len(vecs) >= 1:
            cuisine_protos[cuisine] = store.prototype(vecs)
            print(f"  Created {cuisine} prototype from {len(vecs)} recipes")

    # Test classification of a new recipe query
    test_recipe = {"name": "stir fry noodles", "tags": ["noodles", "quick"]}
    test_vec = store.encoder.encode_data(test_recipe)

    print(f"\n  Classifying: {test_recipe['name']}")
    for cuisine, proto in cuisine_protos.items():
        sim = cosine_similarity(test_vec, proto)
        print(f"    → {cuisine}: {sim:.4f}")

    print("\n🔄 2. DIFFERENCE: Ingredient substitution finder")
    print("-" * 55)

    # Find two similar recipes with different key ingredients
    # e.g., mapo tofu vs butter chicken (both are curried/saucy but different protein)
    tofu_recipe = next((r for r in recipes if "tofu" in r.get("name", "").lower()), None)
    chicken_recipe = next((r for r in recipes if "chicken" in r.get("name", "").lower()), None)

    if tofu_recipe and chicken_recipe:
        tofu_vec = store.encoder.encode_data(tofu_recipe)
        chicken_vec = store.encoder.encode_data(chicken_recipe)

        # Compute what makes chicken different from tofu dish
        diff = store.difference(chicken_vec, tofu_vec)

        print(f"  Comparing: '{tofu_recipe['name']}' vs '{chicken_recipe['name']}'")
        diff_np = diff.cpu().numpy() if hasattr(diff, 'cpu') else diff
        print(f"  Difference vector norm: {np.linalg.norm(diff_np):.1f}")

        # Use difference to find "substitution candidates"
        # Things similar to the difference are "what you'd add to make tofu more like chicken"
        print("\n  Recipes most aligned with the 'tofu→chicken' transformation:")
        scored = []
        for i, recipe in enumerate(recipes):
            vec = store.encoder.encode_data(recipe)
            sim = cosine_similarity(diff, vec)
            scored.append((sim, i, recipe))

        scored.sort(key=lambda x: x[0], reverse=True)
        for sim, _, recipe in scored[:3]:
            print(f"    [{sim:.4f}] {recipe['name']}")

    print("\n🎨 3. BLEND: Create fusion recipe queries")
    print("-" * 55)

    # Blend Italian and Asian cuisine styles
    italian_proto = cuisine_protos.get(":italian")
    asian_proto = cuisine_protos.get(":asian")

    if italian_proto is not None and asian_proto is not None:
        # 50/50 blend of Italian and Asian
        fusion = store.blend(italian_proto, asian_proto, alpha=0.5)

        print("  Blending: 50% Italian + 50% Asian (fusion cuisine)")
        print("\n  Recipes closest to Italian-Asian fusion:")

        scored = []
        for i, recipe in enumerate(recipes):
            vec = store.encoder.encode_data(recipe)
            sim = cosine_similarity(fusion, vec)
            scored.append((sim, i, recipe))

        scored.sort(key=lambda x: x[0], reverse=True)
        for sim, _, recipe in scored[:3]:
            cuisine = recipe.get("cuisine", "?")
            print(f"    [{sim:.4f}] {recipe['name']} ({cuisine})")

    print("\n📢 4. AMPLIFY: Boost health-conscious features")
    print("-" * 55)

    # Create a base query and amplify "healthy" characteristics
    base_query = {"cuisine": ":asian"}
    base_vec = store.encoder.encode_data(base_query)

    healthy_features = {"diet": ["vegan", "gluten-free"], "tags": ["healthy"]}
    healthy_vec = store.encoder.encode_data(healthy_features)

    amplified = store.amplify(base_vec, healthy_vec, strength=3.0)

    print("  Base query: Asian recipes")
    print("  Amplifying: vegan, gluten-free, healthy with strength=3.0")
    print("\n  Top 3 health-amplified Asian matches:")

    scored = []
    for i, recipe in enumerate(recipes):
        vec = store.encoder.encode_data(recipe)
        score = cosine_similarity(amplified, vec)
        scored.append((score, i, recipe))

    scored.sort(key=lambda x: x[0], reverse=True)
    for score, _, recipe in scored[:3]:
        diet = recipe.get("diet", [])
        diet_str = ", ".join(diet) if diet else "none"
        print(f"    [{score:.4f}] {recipe['name']} (diet: {diet_str})")

    print("\n🚫 5. NEGATE: Exclude unwanted characteristics")
    print("-" * 55)

    # Start with comfort food, negate "heavy" dishes
    comfort_query = {"tags": ["comfort"]}
    comfort_vec = store.encoder.encode_data(comfort_query)

    # Negate characteristics of "heavy" dishes (e.g., lasagna, baking)
    heavy_features = {"tags": ["baking", "family"], "time": 90}
    heavy_vec = store.encoder.encode_data(heavy_features)

    negated = store.negate(comfort_vec, heavy_vec)

    print("  Query: Comfort food")
    print("  Negating: baking, family-style, long cooking time")
    print("\n  Light comfort food (after negation):")

    scored = []
    for i, recipe in enumerate(recipes):
        vec = store.encoder.encode_data(recipe)
        score_before = cosine_similarity(comfort_vec, vec)
        score_after = cosine_similarity(negated, vec)
        if "comfort" in recipe.get("tags", []):
            scored.append((score_after, score_before, i, recipe))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        for score_after, score_before, _, recipe in scored[:3]:
            delta = score_after - score_before
            time = recipe.get("time", "?")
            print(f"    [{score_after:.4f}] (Δ{delta:+.4f}) {recipe['name']} ({time}min)")


if __name__ == "__main__":
    main()
