# Kestrel Engineering Handbook - Onboarding

## Your first week
New engineers receive a workstation, a Kestrel single sign-on account, and an
invitation to the Beacon developer portal on day one. Your onboarding buddy adds
you to the team group in the access portal; group membership is what grants
repository and dashboard permissions, so nothing else works until that lands.

## Setting up the toolchain
Install the internal command line tool with `kestrel-cli setup`. It pins the
supported Python and Node versions, configures the package proxy, and writes a
local config file to `~/.kestrel/config.toml`. Run `kestrel-cli doctor` when a
command behaves oddly; it checks tool versions, proxy settings, and token expiry
and prints the first thing it finds wrong.

## Your first change
Every change starts as a branch off `main` and lands as a pull request. A pull
request needs one approving review from a code owner plus a green build before
it can merge. Small pull requests are the norm here: the handbook target is
under four hundred changed lines.

## Code review expectations
Reviewers aim to respond within one business day. Reviews focus on correctness,
tests, and readability. Formatting is never discussed in review because the
pre-commit hook applies it automatically.
