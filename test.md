# Problem: Optimizing Database Queries

## Method 1: Indexing Strategy ⭐ [CHOSEN]
- Pros: Fast, low complexity
- Cons: Storage overhead
  
  <details>
  <summary>Step 2.1: Identify Slow Queries (EXPANDED)</summary>
  
  - Use EXPLAIN ANALYZE
  - Check query execution time > 100ms
  - Tools: pg_stat_statements
  
    <details>
    <summary>Step 2.1.1: Setting up pg_stat_statements</summary>
    
    - Add to postgresql.conf: shared_preload_libraries = 'pg_stat_statements'
    - Result: Found 12 slow queries on users table
    </details>
  </details>

## Method 2: Query Rewriting
- Pros: No schema changes
- Cons: More complex code

## Method 3: Caching Layer
- [collapsed]