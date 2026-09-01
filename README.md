# photo-matcher

Reconciles two product catalogues that describe the same physical products and
share no usable join key, by matching them on what their photographs look like.

```
source_key  source_title          target_key   target_title       best_distance  votes  photos  confidence
SC-0142     Ancient Stone Golem   stone-golem  Stone Golem 32mm   2              4      5      HIGH
SC-0143     Frost Wyrm Large      frost-wyrm   Frost Wyrm         3              3      4      HIGH
SC-0144     Harbour Set B         harbour-b    Harbour Terrain    9              1      3      MEDIUM
SC-0145     Bridge Section        —            —                  —              0      2      NO_MATCH
```

## The problem

Two exports describe the same inventory. They are known to be 1:1. They have
nothing to join on.

This is not a hypothesis — every deterministic key was checked against the real
data and ruled out with evidence *before* anything probabilistic was written:

| Candidate key | Why it failed |
|---|---|
| **Image IDs** | One platform's are in the 8.1-billion range, the other's are unrelated integers. Zero overlap. |
| **SKUs** | Two- or three-letter creator codes on one side, shop and platform prefixes on the other, bare numerics in places. Zero overlap. |
| **Listing IDs** | No source listing ID appears anywhere in the target file. |
| **Titles** | Unreliable — one catalogue's titles were corrupted, which is *why* the reconciliation was needed. Token-Jaccard matching produced confident false positives across different creators. |

That last row is the important one. A fuzzy title match does not fail loudly; it
produces a plausible wrong answer. Wrong matches are worse than missing ones,
because a missing match is visible and a wrong one is not.

What the two catalogues *do* share is the product photography. The same render
appears in both, re-encoded and resized but visually identical.

## Approach

**Perceptual hashing.** A 64-bit pHash is DCT-based, so it survives the things
that break a cryptographic hash — re-encoding, resizing, mild recompression —
and the Hamming distance between two hashes measures visual similarity. The
same render stored at 2000px on one platform and 800px on the other produces
near-identical hashes.

**Then voting, which is what makes the fallback trustworthy.** A listing
usually has several photos. Hash all of them, find each one's nearest neighbour
in the target catalogue, and let the photos vote on which target product the
listing belongs to.

Voting exists because catalogues are full of shared generic images: a
size-comparison shot, a bare base, a brand card reused across hundreds of
listings. A single close hash against one of those drags a match badly off
course — and it drags it *confidently*, at distance 0, because it genuinely is
the same image. Agreement across several photos cannot be manufactured by one
shared image.

Votes are weighted by `1 / (1 + distance)`, so a photo matching at distance 1
counts far more than one scraping in at 20, without any single photo being able
to outvote the others on its own.

## Confidence tiers

Two independent signals: did several photos agree, and is the best single
distance small.

| Tier | Meaning |
|---|---|
| `HIGH` | Multiple photos agree **and** the best distance is very small |
| `MEDIUM` | Agreement **or** a very close single match — but not both |
| `LOW` | A plausible nearest neighbour with weak support. Needs a human |
| `NO_MATCH` | Nothing within the distance ceiling |

A single photo can never reach HIGH, however perfect its distance, precisely
because a shared generic image matches perfectly.

**The LOW and NO_MATCH rows are the deliverable**, as much as the matches. They
are the list a human reviews, and the tiering exists so that review is a
morning's work rather than a re-check of everything.

## Calibration

The thresholds are constants at the top of the file, not buried, and every one
is overridable from the CLI:

```python
HIGH_MAX_DISTANCE = 6        # a re-encoded, resized copy lands within a few bits
MEDIUM_MAX_DISTANCE = 12     # heavier recompression, a crop, a colour shift
DISTANCE_CEILING = 22        # beyond this, treat as unrelated
MIN_VOTES_FOR_AGREEMENT = 2  # one shared image can move one photo, not two
```

`DISTANCE_CEILING` is set from where the distance histogram on real data stopped
being bimodal — matches clustered under ~10, everything else piled up past ~25.
22 sits in the empty middle, so moving it a little changes almost nothing, which
is the property you want from a threshold.

Two random 64-bit hashes differ in 32 bits on average. That is the number to
keep in mind reading any distance here.

## Usage

```bash
pip install -r requirements.txt
python match_by_photo.py --source etsy_export.csv --target shopify_export.xlsx
```

**Use `--limit` on the first run against a new export.** A full pass over tens
of thousands of images is long enough that discovering a parsing bug at the end
of it is expensive:

```bash
python match_by_photo.py --source a.csv --target b.xlsx --limit 20
```

Other flags:

```bash
--out photo_match_mapping.csv   # output path
--cache phash_cache.json        # hash cache; re-runs are near-instant warm
--delay 0.2                     # seconds between fetches
--retry-failures                # re-fetch URLs previously cached as failures
--high-max 6 --medium-max 12 --ceiling 22
```

### Inputs

Both formats, detected by extension:

- **Vela-format CSV** (Etsy side) — parent rows plus variation rows, collapsed
  into their parent before hashing. Photo columns are accepted as `Photo 1`,
  `IMAGE1` or `Image 1`, because that naming difference silently produces a
  file that parses cleanly with zero photos attached.
- **Altera-style XLSX** (Shopify side) — the `Products` sheet, handle-grouped
  rows.

### Speed

Two things make the difference between a job that finishes over lunch and one
that runs overnight:

- **Small renditions are requested directly.** Both CDNs encode size in the URL,
  so the tool rewrites to a ~180px variant rather than downloading a 2000px
  original and discarding 99% of it. Safe because pHash is resize-invariant.
- **Every hash is cached to disk**, keyed on the shrunk URL with Shopify's `?v=`
  cache-buster stripped (it changes when the product is edited, not when the
  image does). A re-run with a warm cache does zero fetches — verified by a
  test, because that is the property the whole tuning workflow depends on.

## Notes and limitations

**This has been run against synthetic fixtures, not a live catalogue.** The
environment it was written in is firewalled from both image CDNs — they return
403 through the egress proxy — so it could not be tested end to end there. That
constraint shaped the design rather than being worked around: all network I/O
sits behind a single function (`fetch_bytes`), and everything else is pure and
tested offline. 67 tests cover both parsers, URL rewriting, Hamming distance,
the voting scheme, confidence bucketing, cache behaviour, and a full
source→target match driven by an injected fetcher.

**Failed fetches are recorded, never skipped silently.** "Matched on 2 of 6
photos" and "matched on 2 of 2" are different claims, and a tool that conflates
them is lying about its own confidence.

**pHash keys on coarse luminance layout.** Two genuinely different products
photographed in the same composition, on the same background, at the same
crop — which is exactly what a disciplined product-photography setup produces —
will sit closer than their visual difference suggests. This is the method's real
weakness, and it is the reason for voting and for the LOW tier rather than a
straight nearest-neighbour assignment. The test suite learned this the hard way:
an early fixture generator varied colour but not composition, and its
"unrelated" images came out 12 bits apart, inside the MEDIUM band.

**Not a deduplicator.** It assumes a 1:1 relationship between catalogues. Given
two source listings that share all their photography, it will happily assign
both to the same target.

**Colour is nearly invisible to it.** A pHash is computed on luminance, so the
same product in two colourways may match at HIGH. Where colourways are distinct
products, the mapping needs a human pass — which is what the tier column is for.

## Licence

MIT — see [LICENSE](LICENSE).
