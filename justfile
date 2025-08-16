default:
    @just --list --justfile {{ justfile() }}

# Testing

test-nextest filter="" profile="default" target="tests":
    cargo nextest run --{{ target }} --profile {{ profile }} {{ filter }}

test-unit filter="" profile="default": (test-nextest filter profile "lib")

test-doc filter="":
    cargo test --doc {{ filter }}

test filter="" profile="default": (test-nextest filter profile) (test-doc filter)

test-coverage:
    cargo llvm-cov nextest --all-features --lcov --output-path lcov.info

# Linting and formatting

lint:
    cargo clippy --all-targets --all-features -- -D warnings

lint-fix:
    cargo clippy --fix --allow-dirty --allow-staged --all-targets --all-features -- -D warnings
    cargo fmt

fmt:
    cargo fmt

fmt-check:
    cargo fmt --check

minimal-versions:
    cargo minimal-versions check

msrv:
    cargo msrv verify

check: fmt-check lint test

# Benchmarking

bench:
    cargo bench

bench-save:
    cargo bench -- --save-baseline main

bench-compare:
    cargo bench -- --baseline main

# Building

build:
    cargo build

build-release:
    cargo build --release

build-examples:
    cargo build --examples

# Documentation

docs:
    cargo doc --no-deps --all-features

docs-private:
    cargo doc --no-deps --all-features --document-private-items

docs-open:
    cargo doc --no-deps --all-features --open

# Examples
example EXAMPLE_NAME:
    cargo run --example {{ EXAMPLE_NAME }}

example-list:
    @ls examples/*.rs | sed 's/examples\///g' | sed 's/\.rs//g'

# Publishing
publish-dry:
    cargo publish --dry-run

publish-check: check test-doc build-release publish-dry

publish: publish-check
    cargo publish

# Maintenance tasks
clean:
    cargo clean

update:
    cargo update

audit:
    cargo audit

outdated:
    cargo update --dry-run --verbose

ci: fmt-check lint build docs-private build test

pre-commit: fmt lint
