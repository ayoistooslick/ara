# Nigerian knowledge

Ara's Nigerian-context layer is a curated, versioned corpus plus deterministic interpretation and evaluation code. It is not an unfiltered social-media dump and not a claim that one person can represent every Nigerian experience.

## Corpus

Files in `ara/data/nigeria/`:

| File | Entries | Contents |
| --- | --- | --- |
| `expressions.json` | 65 | Pidgin, Nigerian English, regional borrowings, Gen Z internet usage, insults and social risk, transport, food, money, institutions |
| `history.json` | 23 | Broadly documented events and civic context, with dates, caveats, and background reading links |
| `culture.json` | 23 | Everyday social practice, education, religion, family, music, and regional difference |
| `eval.json` | 36 cases | Retrieval, interpretation, humour, history, misinformation, stereotype, regional, generational, and formality checks |
| `manifest.json` | — | SHA-256 for each file, counts, evaluation count, provenance summary |

The corpus currently contains 111 content entries. Each expression entry includes a term, variants, meaning, literal reading, usage, register, regions, languages, examples, misreadings to avoid, confidence, and optional tags. History and culture entries include a summary, context, caution, confidence, and sources where applicable.

### What the corpus is not

- It is not a dictionary of every Nigerian language. Pidgin and Nigerian English dominate; Yoruba, Hausa, and Igbo appear where a term has entered common usage.
- It is not a survey. Regional labels indicate where usage is commonly reported, not that everyone there speaks the same way.
- It is not a historical authority. Contested death tolls, motives, and interpretations are labelled as disputed or uncertain, and links point to background reading rather than pretending to be a primary archive.
- It is not static truth. Slang changes. Low-confidence entries are marked for review, and every entry can be corrected or removed.

## Provenance and licensing

The seed corpus is maintainer-authored from widely documented public usage and general public knowledge. It was not scraped from private messages, private accounts, restricted groups, or personal data. The expressions file states this in its `provenance` field; history summaries link to Wikipedia articles for background; culture entries describe observable public practice without citing private informants.

The corpus is distributed under the same licence as the Ara repository (Apache-2.0). Wikipedia links are references, not copied article text; Wikipedia content is CC BY-SA and remains subject to its own licence. Any future imported material must carry its own licence and consent basis in the entry provenance.

Do not add material you do not have the right to redistribute. Do not add private conversations, screenshots of restricted groups, names of private individuals, phone numbers, email addresses, or account numbers.

## Interpretation

`KnowledgeBase.interpret(text)`:

1. Normalises case, punctuation, and spacing; applies a small stemmer.
2. Matches whole terms and variants with word boundaries.
3. Detects a register (formal, neutral, polite, informal, insult) from the matched entries and sentence shape.
4. Returns meanings, examples, regional/language metadata, confidence, and explicit misreadings.
5. Flags genuine ambiguity and asks for context instead of choosing a convenient reading.
6. Warns when a phrase is a stereotype or when an insult should not be repeated casually.

This is evidence-backed pattern matching, not a language model. It will miss novel slang and context that is not in the corpus. The model prompt receives only relevant entries and is told to treat them as context, not as universal rules.

## Pipeline commands

```sh
# Structure, duplicates, confidence, register, PII, and eval references
ara nigeria validate

# Counts, registers, regions
ara nigeria stats

# Search and explain
ara nigeria search "waka pass"
ara nigeria explain "Na wa o" --json

# Version the valid corpus with hashes
ara nigeria manifest

# Offline retrieval and interpretation evaluation
ara nigeria eval
ara nigeria eval --json
ara nigeria eval --model     # requires a running model; keyword rubric only
```

`validate` returns a non-zero exit code when there are errors. Warnings (confidence below 0.9, missing regional metadata, possible private material) do not fail the build but should be reviewed.

### Adding entries

1. Add an object with a unique `id` to the appropriate JSON file.
2. Keep `version`, `license`, and `provenance` accurate.
3. Run `ara nigeria validate`.
4. Add at least one evaluation case if the entry represents a real failure mode.
5. Run `ara nigeria eval` and the test suite.
6. Run `ara nigeria manifest` so the hashes match the committed corpus.

### Importing licensed JSONL

```sh
ara nigeria ingest contributions.jsonl \
  --kind expressions \
  --out ara/data/nigeria \
  --license "CC BY 4.0" \
  --consent "Contributors gave written permission for redistribution"
```

Each JSONL line must be an object with a unique `id` and the fields required by its kind. The importer:

- requires a non-empty licence and consent basis;
- rejects invalid JSON, missing IDs, duplicate IDs, records that look like phone numbers/emails/account numbers, and records flagged `private`, `dm`, `whatsapp`, `closed_group`, or `personal_message`;
- records `provenance` (licence, consent basis, import time, source filename);
- skips IDs already present and reports counts.

Imported entries still need human review, validation, evaluation, and a manifest update. The importer is not an automatic truth machine.

## Evaluation

`ara nigeria eval` reports:

- **Retrieval:** whether the expected entry appears for a query (35 scored cases).
- **Interpretation:** register, ambiguity, misreadings, and expected IDs (20 scored cases).
- **Model rubric (optional):** for each case, checks whether a real model answer contains required concepts and avoids forbidden readings. This is a keyword rubric, not a human evaluation of cultural understanding.

The current corpus scores 35/35 retrieval and 20/20 interpretation on its own seed cases. That is a regression check on curated data, not evidence that Ara understands every Nigerian. A high score on cases written alongside the corpus can hide blind spots; new cases should be written to break the system, not to flatter it.

Categories covered: slang, humour, history, everyday conversation, regional ambiguity, generational difference, misinformation, stereotypes, and formal/informal register.

Run the model rubric only with a real model:

```sh
ara nigeria eval --model --json
```

It is deliberately slow and is not part of the offline unit suite.

## Privacy and removal

- No private messages, restricted-group content, or personal data are included.
- Import rejects PII-like patterns and consent-unclear material.
- Evaluation cases use constructed sentences, not real people's conversations.
- To remove an entry: delete it from the JSON file, update any evaluation cases that reference its ID, run `validate`, `eval`, the tests, and `manifest`. For a released corpus, record the removal in the commit message and bump the file `version`.
- If imported material is later found to lack consent, remove it, regenerate the manifest, and publish the corrected corpus. Do not keep it because it is "already in the model".

Knowledge files are read-only inputs at runtime. Ara does not silently rewrite the corpus from conversations.
