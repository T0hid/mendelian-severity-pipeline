# Name and ordering check

Scenario labels do not feed random seeds. Model tie-breaking uses model names, which are unchanged. Annotation prefixes are added after raw-term hashing; the hashed terms are unchanged.

Feature selection returns columns through a boolean mask in the original input-column order. Family-name sorting changes reporting/iteration order, but family budgets are independent and the final mask follows input order. No feature list or input table is re-sorted by this cleanup.

Saved reference results, fingerprint inventories and the original input archives retain their historical names and values. They are evidence of the previous run, not regenerated outputs. Serialized scorer module aliases and compatibility loading remain available for the bundled model.
