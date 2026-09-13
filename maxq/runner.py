"""Run harness: send each question to each model under identical conditions.

Design (issue #2):
- one adapter per provider (xAI, Anthropic, OpenAI, Google), same prompt template
- fixed temperature, fixed n attempts, full transcripts persisted to results/
- resumable: skip (question, model, attempt) triples already on disk
"""
raise NotImplementedError("See GitHub issue: build the run harness")
