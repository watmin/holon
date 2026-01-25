# Intelligent Bug Report Memory & Duplicate Finder

Goal: Ingest bug reports and quickly find duplicates, similar issues, or related crashes using vector-symbolic geometry.

Bug report fields:
- :id           (string)
- :title        (string – short description)
- :component    (keyword e.g. :ui :backend :auth :mobile)
- :severity     (keyword :critical :high :medium :low)
- :stacktrace   (string – first 3 lines or key frames)
- :environment  (map: {:os keyword :browser keyword :version string})
- :labels       (set of keywords)
- :reported_at  (date string)

Requirements:
1. Ingest 30–60 synthetic bug reports (you can generate them)
2. Support these kinds of queries:
   - "Find all bugs similar to 'crash on login with Google OAuth'" (title + component + env)
   - "High severity bugs in :auth component from last month"
   - "Bugs NOT related to mobile but similar to iOS crash reports" (negation + fuzzy)
   - Cluster similar bugs (group by high similarity threshold)
3. Use Holon to encode both free-text parts (title/stacktrace) and structured fields

Show your VSA encoding strategy especially for text snippets and maps. Demonstrate duplicate detection and a few triage-style queries.
