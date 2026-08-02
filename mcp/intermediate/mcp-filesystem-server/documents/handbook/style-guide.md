# Style guide

Consistency beats cleverness. When two options are equally good, pick the one
that matches the code already around it.

## Naming

- Modules and files use `kebab-case` for directories and `snake_case` for Python.
- Functions say what they do, not how: `resolve_in_sandbox`, not `path_helper2`.
- Booleans read as claims: `is_readonly`, `has_access`.

## Comments

Comments explain *why*, never *what*. If a line needs a comment to say what it
does, rename something instead.

## Errors

Return errors the caller can act on. An error message that only says "invalid
input" costs the reader a debugging session.

## Untrusted paths

Anything that reaches the filesystem from outside the process is untrusted.
Resolve it, then check that the resolved path is still inside the sandbox root
before opening it. A relative path containing `..` and a symlink that points
outside the root are the same bug wearing different hats.

## Reviews

Reviews are for design and correctness. Formatting is the formatter's job.
