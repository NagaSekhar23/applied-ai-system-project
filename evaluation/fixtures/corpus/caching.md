# Caching

A frozen fixture document. Do not edit.

## Cache invalidation

A prefix cache matches from the beginning of the input. Any byte that changes
inside the prefix invalidates everything after it, so volatile values such as
timestamps belong at the end of the input rather than the start.

## Eviction policy

A bounded cache must decide what to discard. Least-recently-used eviction
keeps the working set and discards entries that have gone cold, which suits
workloads where recent items are likely to be requested again.
