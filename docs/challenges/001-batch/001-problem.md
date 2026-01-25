# Personal Task Memory with Fuzzy Retrieval

Goal: Build a lightweight, in-memory "task brain" for a personal assistant that can store tasks with attributes and later retrieve them using fuzzy, partial, negated, or guarded queries.

Task structure (EDN-like / JSON-like):
- :id         (uuid string)
- :title      (string)
- :project    (keyword/string e.g. :work, :personal, :side)
- :priority   (keyword :high :medium :low)
- :due        (date string YYYY-MM-DD or nil)
- :tags       (set of keywords)
- :context    (set of keywords e.g. :computer :phone :errand :home)
- :status     (keyword :todo :waiting :done)

Requirements:
1. Ingest 20–50 example tasks (make up realistic ones)
2. Support queries like:
   - "All high-priority tasks due this week" (guard on due date range)
   - "Tasks NOT in project :work with tag :urgent"
   - "Errand tasks in :home or :phone context"
   - "Tasks whose title is similar to 'prepare taxes' or 'file taxes'"
3. Use Holon's fuzzy similarity, negations, wildcards/guards
4. Show how to get ranked results (top 5 most similar)
5. Bonus: Add one hierarchical query (e.g. project → sub-tasks)

Show full code including imports, encoding strategy, ingestion, and query examples with results.
