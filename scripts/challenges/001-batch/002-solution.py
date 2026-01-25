#!/usr/bin/env python3
"""
Recipe Memory & Substitution Finder using Holon VSA/HDC

This script demonstrates a smart recipe memory system that can store recipes
as structured data and enable similarity search, ingredient substitution,
and advanced querying using Holon's vector symbolic architecture.
"""

import json

from holon import CPUStore


def create_sample_recipes():
    """Generate 12 diverse recipes with varied cuisines, diets, and ingredients."""

    recipes = [
        # Italian recipes
        {
            "name": "Classic Lasagna",
            "cuisine": ":italian",
            "diet": "#{}",  # empty set means no special diet
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
            "tags": '#{"comfort" "family" "baking"}',
        },
        {
            "name": "Vegan Eggplant Parmesan",
            "cuisine": ":italian",
            "diet": '#{"vegan"}',
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
            "tags": '#{"vegan" "italian" "vegetarian"}',
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
            "tags": '#{"thai" "noodles" "quick" "spicy"}',
        },
        {
            "name": "Mapo Tofu",
            "cuisine": ":asian",
            "diet": '#{"vegan"}',
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
            "tags": '#{"chinese" "spicy" "quick" "comfort"}',
        },
        # Mexican recipes
        {
            "name": "Chicken Tacos",
            "cuisine": ":mexican",
            "diet": '#{"gluten-free"}',
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
            "tags": '#{"mexican" "quick" "gluten-free" "protein"}',
        },
        {
            "name": "Vegan Burrito Bowl",
            "cuisine": ":mexican",
            "diet": '#{"vegan" "gluten-free"}',
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
            "tags": '#{"mexican" "vegan" "gluten-free" "healthy"}',
        },
        # Middle Eastern recipes
        {
            "name": "Chicken Shawarma",
            "cuisine": ":middle-eastern",
            "diet": '#{"gluten-free"}',
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
            "tags": '#{"middle-eastern" "grilled" "spicy" "protein"}',
        },
        {
            "name": "Falafel Bowls",
            "cuisine": ":middle-eastern",
            "diet": '#{"vegan" "gluten-free"}',
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
            "tags": '#{"middle-eastern" "vegan" "fried" "healthy"}',
        },
        # Indian recipes
        {
            "name": "Butter Chicken",
            "cuisine": ":indian",
            "diet": '#{"gluten-free"}',
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
            "tags": '#{"indian" "curry" "creamy" "comfort"}',
        },
        {
            "name": "Chana Masala",
            "cuisine": ":indian",
            "diet": '#{"vegan" "gluten-free"}',
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
            "tags": '#{"indian" "curry" "vegan" "protein"}',
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
            "tags": '#{"american" "quick" "comfort" "sandwich"}',
        },
        {
            "name": "Vegan Mac and Cheese",
            "cuisine": ":american",
            "diet": '#{"vegan"}',
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
            "tags": '#{"american" "vegan" "comfort" "pasta"}',
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
            edn_parts.append(f":{key} {value}")
        elif key == "difficulty":
            edn_parts.append(f":{key} {value}")
        elif key == "time":
            edn_parts.append(f":{key} {value}")
        elif key == "ingredients":
            edn_parts.append(
                f':{key} [{", ".join(format_ingredient(ing) for ing in value)}]'
            )
        elif key == "tags":
            edn_parts.append(f":{key} {value}")

    return f"{{{', '.join(edn_parts)}}}"


def ingest_recipes(store, recipes):
    """Ingest recipes into the Holon store."""
    print(f"📥 Ingesting {len(recipes)} recipes into Holon memory...")

    for i, recipe in enumerate(recipes):
        # Convert to EDN format
        recipe_edn = convert_recipe_to_edn(recipe)
        recipe_id = store.insert(recipe_edn, data_type="edn")
        if (i + 1) % 3 == 0:
            print(f"  ✓ Ingested {i + 1}/{len(recipes)} recipes")

    print("✅ All recipes ingested successfully!")


def query_recipes(
    store, query, description, top_k=10, guard=None, negations=None, data_type="edn"
):
    """Query recipes and display results."""
    print(f"\n🔍 {description}")
    print(f"Query: {query}")
    if guard:
        print(f"Guard: {guard}")
    if negations:
        print(f"Negations: {negations}")

    try:
        results = store.query(
            query,
            data_type=data_type,
            guard=guard,
            negations=negations,
            top_k=top_k,
            threshold=0.0,
        )

        if not results:
            print("  ❌ No matching recipes found")
            return

        print(
            f"  ✅ Found {len(results)} matching recipes (showing top {min(top_k, len(results))}):"
        )

        for i, (recipe_id, score, recipe_data) in enumerate(results):
            recipe = recipe_data  # Already parsed EDN
            # EDN keywords are parsed as Keyword objects, need to access by keyword
            from edn_format import Keyword

            name_key = Keyword("name")
            cuisine_key = Keyword("cuisine")
            difficulty_key = Keyword("difficulty")
            time_key = Keyword("time")
            diet_key = Keyword("diet")
            tags_key = Keyword("tags")

            print(f"  {i+1}. [{score:.3f}] {recipe[name_key]}")
            print(
                f"     Cuisine: {recipe[cuisine_key]} | Difficulty: {recipe[difficulty_key]} | Time: {recipe[time_key]} min"
            )
            if recipe.get(diet_key) and str(recipe[diet_key]) != "#{}":
                print(f"     Diet: {recipe[diet_key]}")
            if recipe.get(tags_key) and str(recipe[tags_key]) != "#{}":
                print(f"     Tags: {recipe[tags_key]}")

    except Exception as e:
        print(f"  ❌ Query failed: {e}")


def main():
    """Main demonstration function."""
    print("🍳 Recipe Memory & Substitution Finder Demo")
    print("=" * 55)

    # Initialize Holon store
    print("🚀 Initializing Holon CPUStore...")
    store = CPUStore(dimensions=16000)
    print("✅ Store initialized with 16,000 dimensions")

    # Create and ingest sample recipes
    recipes = create_sample_recipes()
    ingest_recipes(store, recipes)

    # Demonstrate various query types
    print("\n" + "=" * 55)
    print("🧪 QUERY DEMONSTRATIONS")
    print("=" * 55)

    # 1. Find recipes similar to "classic lasagna"
    query_recipes(
        store,
        '{:name "classic lasagna"}',
        "1. FUZZY SIMILARITY: Recipes similar to 'classic lasagna'",
    )

    # 2. Recipes similar to pad thai, but without shrimp
    query_recipes(
        store,
        '{:name "pad thai"}',
        "2. SIMILARITY + NEGATION: Pad thai similar recipes, no shrimp",
        negations={"ingredients": [{"item": "shrimp"}]},
    )

    # 3. What can replace tofu in mapo tofu recipe? (find structurally similar dishes with different proteins)
    query_recipes(
        store,
        '{:name "mapo tofu"}',
        "3. SUBSTITUTION: Structurally similar to mapo tofu but with different main protein",
        negations={"ingredients": [{"item": "tofu"}]},
    )

    # 4. Dishes with "curry" in tags
    query_recipes(
        store, '{:tags #{"curry"}}', "4. TAG SIMILARITY: Dishes with 'curry' tag"
    )

    # 5. Asian cuisine recipes
    query_recipes(store, "{:cuisine :asian}", "5. CUISINE FILTER: Asian recipes")

    # 6. Vegan recipes
    query_recipes(store, '{:diet #{"vegan"}}', "6. DIET FILTER: Vegan recipes")

    # 7. Comfort food recipes
    query_recipes(
        store, '{:tags #{"comfort"}}', "7. TAG SIMILARITY: Comfort food recipes"
    )

    print("\n" + "=" * 55)
    print("🎉 Recipe Memory Demo Complete!")
    print(
        "Holon successfully demonstrated recipe similarity, substitution, and advanced querying"
    )
    print("=" * 55)


if __name__ == "__main__":
    main()
