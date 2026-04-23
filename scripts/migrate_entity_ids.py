#!/usr/bin/env python3
"""
Migrate existing Neo4j entities to stable canonical IDs.

Reads all :Entity nodes that lack an `id` property, computes the
canonical ID from (name, type), merges duplicates, and writes back.

Usage:
    python scripts/migrate_entity_ids.py          # dry-run (prints plan)
    python scripts/migrate_entity_ids.py --apply  # execute writes
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entity_canonicalizer import canonical_id


def run(apply: bool = False):
    from neo4j import GraphDatabase

    uri      = os.environ["NEO4J_URI"]
    user     = os.environ["NEO4J_USER"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session(database=database) as session:
        # Fetch all entities without a stable id
        records = session.run(
            "MATCH (e:Entity) WHERE e.id IS NULL RETURN e.name AS name, e.type AS type, elementId(e) AS eid"
        ).data()

        print(f"Found {len(records)} entities without canonical id")

        seen: dict[str, list] = {}
        for r in records:
            cid = canonical_id(r["name"] or "", r["type"] or "UNKNOWN")
            seen.setdefault(cid, []).append(r)

        merges = [(cid, group) for cid, group in seen.items() if len(group) > 1]
        updates = [(cid, group[0]) for cid, group in seen.items() if len(group) == 1]

        print(f"  {len(updates)} entities will get id assigned")
        print(f"  {len(merges)} canonical-id groups have duplicates to merge")

        if not apply:
            print("\nDry-run complete. Pass --apply to execute.")
            driver.close()
            return

        # Assign ids to singletons
        for cid, r in updates:
            session.run(
                "MATCH (e:Entity) WHERE elementId(e) = $eid SET e.id = $id",
                eid=r["eid"], id=cid,
            )

        # Merge duplicate groups: keep first, re-point relationships, delete rest
        for cid, group in merges:
            primary_eid = group[0]["eid"]
            session.run(
                "MATCH (e:Entity) WHERE elementId(e) = $eid SET e.id = $id",
                eid=primary_eid, id=cid,
            )
            for dup in group[1:]:
                dup_eid = dup["eid"]
                # Re-point outgoing relationships
                session.run("""
                    MATCH (dup:Entity)-[r]->(other)
                    WHERE elementId(dup) = $dup_eid
                    MATCH (primary:Entity) WHERE elementId(primary) = $primary_eid
                    MERGE (primary)-[r2:RELATES_TO]->(other)
                    SET r2 = properties(r)
                    DELETE r
                """, dup_eid=dup_eid, primary_eid=primary_eid)
                # Re-point incoming relationships
                session.run("""
                    MATCH (other)-[r]->(dup:Entity)
                    WHERE elementId(dup) = $dup_eid
                    MATCH (primary:Entity) WHERE elementId(primary) = $primary_eid
                    MERGE (other)-[r2:RELATES_TO]->(primary)
                    SET r2 = properties(r)
                    DELETE r
                """, dup_eid=dup_eid, primary_eid=primary_eid)
                session.run(
                    "MATCH (e:Entity) WHERE elementId(e) = $eid DETACH DELETE e",
                    eid=dup_eid,
                )

        print("Migration applied.")

    driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Execute writes (default: dry-run)")
    args = parser.parse_args()
    run(apply=args.apply)
