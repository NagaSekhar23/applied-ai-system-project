# Testing

A frozen fixture document. Do not edit.

## Flaky tests

A test that passes and fails without any code change destroys trust in the
suite. The usual causes are shared mutable state between tests, reliance on
wall-clock timing, and unseeded randomness.

## Fixtures and isolation

Each test should construct the state it needs and leave nothing behind.
Isolation is what makes a failure mean something: if tests can affect each
other, a red result no longer identifies which change broke what.
