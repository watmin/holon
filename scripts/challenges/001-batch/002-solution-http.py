#!/usr/bin/env python3
"""
Recipe Memory & Substitution Finder using Holon HTTP API

This script demonstrates a smart recipe memory system that can store recipes
as structured data and enable similarity search, ingredient substitution,
and advanced querying using Holon's HTTP API (network service).
"""

import json
from typing import Any, Dict, List

import requests

# HTTP API Configuration
BASE_URL = "http://localhost:8000"


def http_insert(data: str, data_type: str = "edn") -> str:
    """Insert data via HTTP API."""
    response = requests.post(
        f"{BASE_URL}/insert", json={"data": data, "data_type": data_type}
    )
    response.raise_for_status()
    return response.json()["id"]


def http_batch_insert(items: List[str], data_type: str = "edn") -> List[str]:
    """Batch insert data via HTTP API."""
    response = requests.post(
        f"{BASE_URL}/batch_insert", json={"items": items, "data_type": data_type}
    )
    response.raise_for_status()
    return response.json()["ids"]


def http_query(
    probe: str,
    data_type: str = "edn",
    top_k: int = 10,
    threshold: float = 0.0,
    guard: str = None,
    negations: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """Query data via HTTP API."""
    query_data = {
        "probe": probe,
        "data_type": data_type,
        "top_k": top_k,
        "threshold": threshold,
    }
    if guard:
        query_data["guard"] = guard
    if negations:
        query_data["negations"] = negations

    response = requests.post(f"{BASE_URL}/query", json=query_data)
    response.raise_for_status()
    return response.json()["results"]


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


def ingest_recipes_http(recipes):
    """Ingest recipes via HTTP API."""
    print(f"📥 Ingesting {len(recipes)} recipes via HTTP API...")

    # Convert all recipes to EDN strings
    edn_recipes = [convert_recipe_to_edn(recipe) for recipe in recipes]

    # Batch insert via HTTP
    try:
        ids = http_batch_insert(edn_recipes, "edn")
        print(f"✅ Successfully ingested {len(ids)} recipes via HTTP API")
        return ids
    except Exception as e:
        print(f"❌ HTTP batch insert failed: {e}")
        return []


def query_recipes_http(probe, description, top_k=10, guard=None, negations=None):
    """Query recipes via HTTP API."""
    print(f"\n🔍 {description}")
    print(f"Query: {probe}")
    if guard:
        print(f"Guard: {guard}")
    if negations:
        print(f"Negations: {negations}")

    try:
        results = http_query(
            probe, "edn", top_k=top_k, guard=guard, negations=negations
        )

        if not results:
            print("  ❌ No matching recipes found")
            return

        print(
            f"  ✅ Found {len(results)} matching recipes (showing top {min(top_k, len(results))}):"
        )

        for i, result in enumerate(results):
            recipe = result["data"]
            print(f"  {i+1}. [{result['score']:.3f}] {recipe.get('name', 'Unknown')}")
            print(
                f"     Cuisine: {recipe.get('cuisine', 'Unknown')} | Difficulty: {recipe.get('difficulty', 'Unknown')} | Time: {recipe.get('time', 'Unknown')} min"
            )
            if (
                recipe.get("diet") and recipe["diet"] != []
            ):  # EDN sets become lists in JSON
                print(f"     Diet: {recipe['diet']}")
            if recipe.get("tags") and recipe["tags"] != []:
                print(f"     Tags: {recipe['tags']}")

    except Exception as e:
        print(f"  ❌ HTTP query failed: {e}")


def main():
    """Main demonstration function."""
    print("🍳 Recipe Memory & Substitution Finder (HTTP API)")
    print("=" * 60)

    # Test server connection
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        health = response.json()
        print(f"🔗 Connected to Holon service at {BASE_URL}")
        print(
            f"   Status: {health['status']} | Backend: {health['backend']} | Items: {health['items_count']}"
        )
    except Exception as e:
        print(f"❌ Cannot connect to Holon service: {e}")
        print("Make sure the server is running: python scripts/server/holon_server.py")
        return

    # Create and ingest sample recipes via HTTP
    recipes = create_sample_recipes()
    ids = ingest_recipes_http(recipes)
    if not ids:
        print("❌ Failed to ingest recipes, aborting demo")
        return

    # Demonstrate various query types via HTTP
    print("\n" + "=" * 60)
    print("🧪 HTTP QUERY DEMONSTRATIONS")
    print("=" * 60)

    # 1. Fuzzy similarity query
    query_recipes_http(
        '{:name "classic lasagna"}',
        "1. FUZZY SIMILARITY: Recipes similar to 'classic lasagna'",
    )

    # 2. Similarity + negation (no shrimp)
    query_recipes_http(
        '{:name "pad thai"}',
        "2. SIMILARITY + NEGATION: Pad thai similar recipes, no shrimp",
        negations={"ingredients": [{"item": "shrimp"}]},
    )

    # 3. Substitution query (different proteins from mapo tofu)
    query_recipes_http(
        '{:name "mapo tofu"}',
        "3. SUBSTITUTION: Structurally similar to mapo tofu but with different main protein",
        negations={"ingredients": [{"item": "tofu"}]},
    )

    # 4. Tag-based query (curry dishes)
    query_recipes_http(
        '{:tags #{"curry"}}', "4. TAG SIMILARITY: Dishes with 'curry' tag"
    )

    # 5. Cuisine filter (Asian recipes)
    query_recipes_http("{:cuisine :asian}", "5. CUISINE FILTER: Asian recipes")

    # 6. Diet filter (vegan recipes)
    query_recipes_http('{:diet #{"vegan"}}', "6. DIET FILTER: Vegan recipes")

    # 7. Comfort food query
    query_recipes_http(
        '{:tags #{"comfort"}}', "7. TAG SIMILARITY: Comfort food recipes"
    )

    print("\n" + "=" * 60)
    print("🎉 HTTP Recipe Memory Demo Complete!")
    print(
        "Holon HTTP API successfully demonstrated recipe similarity, substitution, and advanced querying"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
