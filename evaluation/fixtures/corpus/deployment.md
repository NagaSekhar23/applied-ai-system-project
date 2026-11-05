# Deployment

A frozen fixture document. Do not edit.

## Rollout strategy

A canary release sends a small share of traffic to the new version while the
rest continues to the old one. If error rates rise on the canary, the rollout
halts before most users are affected.

## Health checks

A readiness probe reports whether an instance can accept traffic yet. A
liveness probe reports whether it has wedged and should be restarted. Confusing
the two causes restart loops during slow startup.
