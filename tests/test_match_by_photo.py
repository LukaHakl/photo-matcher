"""Tests for the photo matcher. No network is touched anywhere in this file.

The one function that would use it, ``fetch_bytes``, is injected as ``fetcher``
so the whole pipeline can be driven end to end from in-memory PNGs.
"""

from __future__ import annotations

import csv
import io
import json
import random
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from match_by_photo import (  # noqa: E402
    HIGH, LOW, MEDIUM, NO_MATCH, HashCache, Listing, bucket, build_target_index,
    cache_key, hamming, hash_listings, load_catalogue, match_catalogues,
    parse_altera_rows, parse_vela_csv, phash_bits, photo_columns,
    smallest_variant, summarise, vote, write_mapping,
)


# ---------------------------------------------------------------------------
# Image helpers -- synthetic, deterministic, no network
# ---------------------------------------------------------------------------

def make_image(seed: int, size: int = 400) -> Image.Image:
    """A deterministic, visually distinctive image.

    Needs real structure: pHash measures low-frequency DCT content, so a flat
    or purely random field produces degenerate hashes and the tests would pass
    for the wrong reason.

    It also needs real *variety* between seeds. An earlier version varied only
    the colours and offsets of a fixed layout, and the closest pair of
    supposedly-unrelated images came out 12 bits apart -- inside the MEDIUM
    band, which made `test_a_source_with_no_twin_comes_back_no_match` fail for
    an entirely legitimate reason. pHash keys on coarse luminance layout, so
    two images with the same composition are genuinely similar no matter what
    colours they are painted in. The seed therefore drives the *structure*:
    background brightness, how many blocks there are, and where the mass sits.
    """
    rng = random.Random(seed)
    background = rng.randint(0, 3) * 70
    image = Image.new("RGB", (size, size), (background, background, background))
    draw = ImageDraw.Draw(image)

    for _ in range(rng.randint(3, 9)):
        x0, y0 = rng.randrange(size), rng.randrange(size)
        x1 = x0 + rng.randrange(size // 6, size // 2)
        y1 = y0 + rng.randrange(size // 6, size // 2)
        colour = tuple(rng.randrange(256) for _ in range(3))
        if rng.random() < 0.5:
            draw.rectangle([x0, y0, x1, y1], fill=colour)
        else:
            draw.ellipse([x0, y0, x1, y1], fill=colour)
    return image


def encode(image: Image.Image, fmt: str = "PNG", **kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# URL shrinking
# ---------------------------------------------------------------------------

def test_etsy_url_is_shrunk_in_place():
    url = "https://i.etsystatic.com/12345/r/il/ab12cd/987654321/il_fullxfull.987654321_x1y2.jpg"
    assert "il_180xN" in smallest_variant(url)
    assert "il_fullxfull" not in smallest_variant(url)


@pytest.mark.parametrize("token", ["il_fullxfull", "il_1588xN", "il_75x75", "il_340x270"])
def test_every_etsy_rendition_token_is_recognised(token):
    url = "https://i.etsystatic.com/1/r/il/a/2/%s.9_ab.jpg" % token
    assert "il_180xN" in smallest_variant(url)


def test_shopify_width_parameter_is_rewritten():
    url = "https://cdn.shopify.com/s/files/1/0/1/products/thing.jpg?v=1699&width=2048"
    assert "width=180" in smallest_variant(url)


def test_shopify_dimension_suffix_is_rewritten():
    url = "https://cdn.shopify.com/s/files/1/0/1/products/thing_1024x1024.jpg?v=17"
    out = smallest_variant(url)
    assert "_180x" in out and "1024x1024" not in out


def test_unknown_cdn_is_left_alone():
    """An unrecognised host costs bandwidth, never correctness."""
    url = "https://images.example.net/a/b/photo.jpg"
    assert smallest_variant(url) == url


def test_cache_key_drops_the_shopify_cache_buster():
    """?v= changes when the product is edited, not when the image changes."""
    a = "https://cdn.shopify.com/s/files/1/0/1/products/x_180x.jpg?v=111"
    b = "https://cdn.shopify.com/s/files/1/0/1/products/x_180x.jpg?v=999"
    assert cache_key(a) == cache_key(b)


def test_cache_key_is_computed_on_the_shrunk_url():
    """Otherwise a cache built at one size feeds fetches at another."""
    url = "https://i.etsystatic.com/1/r/il/a/2/il_fullxfull.9_ab.jpg"
    assert "il_180xN" in cache_key(url)


# ---------------------------------------------------------------------------
# Photo column detection -- brief 01's silent-failure trap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spelling", ["Photo 1", "IMAGE1", "Image 1", "photo_1"])
def test_all_photo_column_spellings_are_accepted(spelling):
    """Vela imports 'Photo 1'; Etsy exports 'IMAGE1'. Reading the wrong one
    yields a file that parses cleanly with zero photos attached."""
    assert photo_columns(["TITLE", spelling]) == [spelling]


def test_photo_columns_come_back_in_slot_order_not_file_order():
    header = ["TITLE", "IMAGE10", "IMAGE2", "IMAGE1"]
    assert photo_columns(header) == ["IMAGE1", "IMAGE2", "IMAGE10"]


def test_non_photo_columns_are_ignored():
    assert photo_columns(["TITLE", "PRICE", "SKU", "VARIATION 1 TYPE"]) == []


# ---------------------------------------------------------------------------
# Vela CSV parsing
# ---------------------------------------------------------------------------

def vela_csv(rows: list[dict]) -> io.StringIO:
    fields = ["TITLE", "SKU", "Photo 1", "Photo 2", "Photo 3"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({f: row.get(f, "") for f in fields})
    buffer.seek(0)
    return buffer


def test_simple_listing_is_parsed():
    listings = parse_vela_csv(vela_csv([
        {"TITLE": "Dragon", "SKU": "AC-dragon", "Photo 1": "a.jpg", "Photo 2": "b.jpg"},
    ]))
    assert list(listings) == ["AC-dragon"]
    assert listings["AC-dragon"].images == ["a.jpg", "b.jpg"]


def test_variation_rows_collapse_into_their_parent():
    """Child rows leave TITLE empty and belong to the row above."""
    listings = parse_vela_csv(vela_csv([
        {"TITLE": "Dragon", "SKU": "AC-dragon", "Photo 1": "a.jpg"},
        {"Photo 1": "b.jpg"},
        {"Photo 1": "c.jpg"},
        {"TITLE": "Golem", "SKU": "AC-golem", "Photo 1": "d.jpg"},
    ]))
    assert len(listings) == 2
    assert listings["AC-dragon"].images == ["a.jpg", "b.jpg", "c.jpg"]
    assert listings["AC-golem"].images == ["d.jpg"]


def test_duplicate_images_across_variation_rows_are_deduplicated():
    listings = parse_vela_csv(vela_csv([
        {"TITLE": "Dragon", "SKU": "S1", "Photo 1": "a.jpg"},
        {"Photo 1": "a.jpg"},
        {"Photo 1": "b.jpg"},
    ]))
    assert listings["S1"].images == ["a.jpg", "b.jpg"]


def test_title_is_the_key_when_sku_is_missing():
    listings = parse_vela_csv(vela_csv([{"TITLE": "No SKU Here", "Photo 1": "a.jpg"}]))
    assert list(listings) == ["No SKU Here"]


def test_child_rows_before_any_parent_are_skipped_not_invented():
    listings = parse_vela_csv(vela_csv([
        {"Photo 1": "orphan.jpg"},
        {"TITLE": "Real", "SKU": "S1", "Photo 1": "a.jpg"},
    ]))
    assert list(listings) == ["S1"]
    assert "orphan.jpg" not in listings["S1"].images


def test_a_repeated_key_merges_rather_than_clobbers():
    listings = parse_vela_csv(vela_csv([
        {"TITLE": "Dragon", "SKU": "S1", "Photo 1": "a.jpg"},
        {"TITLE": "Dragon", "SKU": "S1", "Photo 1": "b.jpg"},
    ]))
    assert listings["S1"].images == ["a.jpg", "b.jpg"]


def test_empty_file_parses_to_nothing():
    assert parse_vela_csv(io.StringIO("")) == {}


def test_photos_are_capped_at_ten():
    fields = ["TITLE", "SKU"] + ["Photo %d" % i for i in range(1, 13)]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    row = {"TITLE": "T", "SKU": "S"}
    row.update({"Photo %d" % i: "img%d.jpg" % i for i in range(1, 13)})
    writer.writerow(row)
    buffer.seek(0)
    assert len(parse_vela_csv(buffer)["S"].images) == 10


# ---------------------------------------------------------------------------
# Altera row parsing
# ---------------------------------------------------------------------------

def test_handle_grouped_rows_collect_their_images():
    listings = parse_altera_rows([
        {"Handle": "dragon", "Title": "Dragon", "Image Src": "a.jpg"},
        {"Handle": "dragon", "Title": "", "Image Src": "b.jpg"},
        {"Handle": "golem", "Title": "Golem", "Image Src": "c.jpg"},
    ])
    assert listings["dragon"].images == ["a.jpg", "b.jpg"]
    assert listings["dragon"].title == "Dragon"
    assert listings["golem"].images == ["c.jpg"]


def test_blank_handle_continues_the_previous_block():
    listings = parse_altera_rows([
        {"Handle": "dragon", "Title": "Dragon", "Image Src": "a.jpg"},
        {"Handle": "", "Title": "", "Image Src": "b.jpg"},
    ])
    assert listings["dragon"].images == ["a.jpg", "b.jpg"]


def test_title_on_a_later_row_is_not_lost():
    listings = parse_altera_rows([
        {"Handle": "dragon", "Image Src": "a.jpg"},
        {"Handle": "dragon", "Title": "Dragon", "Image Src": "b.jpg"},
    ])
    assert listings["dragon"].title == "Dragon"


def test_rows_with_no_image_still_register_the_product():
    listings = parse_altera_rows([{"Handle": "dragon", "Title": "Dragon"}])
    assert listings["dragon"].images == []


# ---------------------------------------------------------------------------
# Hamming distance
# ---------------------------------------------------------------------------

def test_identical_hashes_are_distance_zero():
    assert hamming(0xDEADBEEFCAFEF00D, 0xDEADBEEFCAFEF00D) == 0


def test_distance_counts_differing_bits():
    assert hamming(0b1010, 0b0101) == 4
    assert hamming(0, 0xFFFFFFFFFFFFFFFF) == 64


def test_distance_is_symmetric():
    assert hamming(0x1234, 0xABCD) == hamming(0xABCD, 0x1234)


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def test_agreement_plus_a_close_distance_is_high():
    assert bucket(best_distance=2, votes=3) == HIGH


def test_agreement_with_a_loose_distance_is_only_medium():
    assert bucket(best_distance=15, votes=3) == MEDIUM


def test_a_very_close_single_photo_is_medium_not_high():
    """One photo cannot be HIGH: a shared generic image matches perfectly."""
    assert bucket(best_distance=1, votes=1) == MEDIUM


def test_weak_support_and_a_loose_distance_is_low():
    assert bucket(best_distance=18, votes=1) == LOW


def test_beyond_the_ceiling_is_no_match():
    assert bucket(best_distance=40, votes=2) == NO_MATCH


def test_zero_votes_is_no_match():
    assert bucket(best_distance=0, votes=0) == NO_MATCH


def test_thresholds_are_overridable():
    assert bucket(best_distance=8, votes=2) == MEDIUM
    assert bucket(best_distance=8, votes=2, high_max=10) == HIGH


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------

def test_photos_agreeing_beats_a_single_closer_photo():
    """The core reason voting exists.

    'decoy' is a shared generic image that matches one source photo exactly --
    a size chart, say. 'real' matches two photos at a slightly worse distance.
    Agreement must win, or every listing carrying the generic shot collapses
    onto whichever product happens to own it.
    """
    index = [
        (0b0000, "decoy"),
        (0b0011, "real"),
        (0b0101, "real"),
    ]
    winner, distance, votes = vote([0b0000, 0b0011, 0b0101], index)
    assert winner == "real"
    assert votes == 2
    assert distance == 0


def test_a_lone_exact_match_still_wins_when_nothing_disagrees():
    index = [(0b0000, "only"), (0xFFFFFFFF, "far")]
    winner, distance, votes = vote([0b0000], index)
    assert (winner, distance, votes) == ("only", 0, 1)


def test_photos_beyond_the_ceiling_abstain():
    """A listing with no twin must return nothing, not the least-bad option."""
    index = [(0xFFFFFFFFFFFFFFFF, "unrelated")]
    winner, _distance, votes = vote([0x0000000000000000], index)
    assert winner is None and votes == 0


def test_no_photos_yields_no_winner():
    assert vote([], [(0, "a")])[0] is None


def test_empty_index_yields_no_winner():
    assert vote([0b1010], [])[0] is None


def test_closer_distance_breaks_a_weight_tie():
    index = [(0b0001, "near"), (0b0111, "far")]
    winner, _d, _v = vote([0b0001, 0b0111], index)
    assert winner == "near"


def test_build_target_index_flattens_every_photo():
    index = build_target_index({"a": [1, 2], "b": [3]})
    assert sorted(index) == [(1, "a"), (2, "a"), (3, "b")]


# ---------------------------------------------------------------------------
# Perceptual hashing: the property the whole method rests on
# ---------------------------------------------------------------------------

def test_a_resized_and_reencoded_copy_still_matches_at_high():
    """The acceptance criterion from the brief.

    The same render, re-encoded PNG->JPEG and resized 400px->180px, is what the
    two catalogues actually store. If this does not land in HIGH, the method
    does not work.
    """
    original = make_image(seed=7, size=400)
    reencoded = original.resize((180, 180), Image.LANCZOS)

    a = phash_bits(encode(original, "PNG"))
    b = phash_bits(encode(reencoded, "JPEG", quality=72))

    distance = hamming(a, b)
    assert distance <= 6, "distance %d is too high for the same image" % distance
    assert bucket(distance, votes=2) == HIGH


def test_different_images_are_far_apart():
    a = phash_bits(encode(make_image(seed=1)))
    b = phash_bits(encode(make_image(seed=99)))
    assert hamming(a, b) > 6


def test_hash_is_stable_across_runs():
    data = encode(make_image(seed=3))
    assert phash_bits(data) == phash_bits(data)


def test_palettised_and_greyscale_images_do_not_crash():
    """Real catalogues contain both; Pillow will not convert implicitly."""
    for mode in ("P", "L"):
        converted = make_image(seed=5).convert(mode)
        assert isinstance(phash_bits(encode(converted, "PNG")), int)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def test_cache_round_trips_through_disk(tmp_path):
    path = str(tmp_path / "c.json")
    cache = HashCache(path)
    cache.put("u1", 0xDEADBEEFCAFEF00D)
    cache.save()

    found, value = HashCache(path).get("u1")
    assert found and value == 0xDEADBEEFCAFEF00D


def test_failures_are_cached_so_they_are_not_refetched(tmp_path):
    path = str(tmp_path / "c.json")
    cache = HashCache(path)
    cache.put("bad", None)
    cache.save()

    found, value = HashCache(path).get("bad")
    assert found is True and value is None


def test_retry_failures_reopens_only_the_failures(tmp_path):
    path = str(tmp_path / "c.json")
    cache = HashCache(path)
    cache.put("bad", None)
    cache.put("good", 5)
    cache.save()

    retry = HashCache(path, retry_failures=True)
    assert retry.get("bad")[0] is False
    assert retry.get("good") == (True, 5)


def test_a_corrupt_cache_file_does_not_crash_the_run(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{not json", encoding="utf-8")
    assert HashCache(str(path)).entries == {}


def test_saving_is_atomic_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "c.json"
    cache = HashCache(str(path))
    cache.put("u", 1)
    cache.save()
    assert path.exists() and not (tmp_path / "c.json.tmp").exists()
    assert json.loads(path.read_text())["u"] == "0000000000000001"


# ---------------------------------------------------------------------------
# hash_listings, with an injected fetcher instead of the network
# ---------------------------------------------------------------------------

def make_fetcher(mapping, calls):
    def fetcher(url, session=None):
        calls.append(url)
        if url not in mapping:
            raise OSError("404 %s" % url)
        return mapping[url]
    return fetcher


def test_a_warm_cache_does_zero_fetches(tmp_path):
    """The acceptance criterion: re-running with a warm cache is free."""
    images = {"http://x/a.png": encode(make_image(1)),
              "http://x/b.png": encode(make_image(2))}
    listings = {"p1": Listing("p1", "P1", ["http://x/a.png", "http://x/b.png"])}
    path = str(tmp_path / "c.json")

    first_calls: list[str] = []
    hash_listings(listings, HashCache(path),
                  fetcher=make_fetcher(images, first_calls))
    assert len(first_calls) == 2

    second_calls: list[str] = []
    hashes, failures = hash_listings(listings, HashCache(path),
                                     fetcher=make_fetcher(images, second_calls))
    assert second_calls == []
    assert failures == []
    assert len(hashes["p1"]) == 2


def test_a_failed_fetch_is_recorded_not_silently_skipped(tmp_path):
    """'Matched 2 of 6 photos' and 'matched 2 of 2' are different claims."""
    images = {"http://x/a.png": encode(make_image(1))}
    listings = {"p1": Listing("p1", "P1", ["http://x/a.png", "http://x/missing.png"])}

    hashes, failures = hash_listings(listings, HashCache(str(tmp_path / "c.json")),
                                     fetcher=make_fetcher(images, []))
    assert len(hashes["p1"]) == 1
    assert failures == ["http://x/missing.png"]


def test_one_bad_image_does_not_end_the_run(tmp_path):
    images = {"http://x/a.png": encode(make_image(1)),
              "http://x/c.png": encode(make_image(3))}
    listings = {
        "p1": Listing("p1", "P1", ["http://x/a.png"]),
        "p2": Listing("p2", "P2", ["http://x/broken.png"]),
        "p3": Listing("p3", "P3", ["http://x/c.png"]),
    }
    hashes, failures = hash_listings(listings, HashCache(str(tmp_path / "c.json")),
                                     fetcher=make_fetcher(images, []))
    assert len(hashes["p1"]) == 1 and len(hashes["p3"]) == 1
    assert hashes["p2"] == [] and len(failures) == 1


# ---------------------------------------------------------------------------
# End to end, still with no network
# ---------------------------------------------------------------------------

def test_end_to_end_matches_the_right_twins(tmp_path):
    """Three products, each appearing in both catalogues as a resized JPEG."""
    source_images, target_images = {}, {}
    source, target = {}, {}

    for i in (1, 2, 3):
        original = make_image(seed=i * 11, size=400)
        twin = original.resize((180, 180), Image.LANCZOS)

        source_url = "http://etsy/%d.png" % i
        target_url = "http://shop/%d.jpg" % i
        source_images[source_url] = encode(original, "PNG")
        target_images[target_url] = encode(twin, "JPEG", quality=75)

        source["S%d" % i] = Listing("S%d" % i, "Source %d" % i, [source_url])
        target["T%d" % i] = Listing("T%d" % i, "Target %d" % i, [target_url])

    everything = {**source_images, **target_images}
    cache = HashCache(str(tmp_path / "c.json"))
    target_hashes, _ = hash_listings(target, cache, fetcher=make_fetcher(everything, []))
    source_hashes, _ = hash_listings(source, cache, fetcher=make_fetcher(everything, []))

    matches = match_catalogues(source, target, source_hashes, target_hashes)
    pairs = {m.source_key: m.target_key for m in matches}
    assert pairs == {"S1": "T1", "S2": "T2", "S3": "T3"}
    assert all(m.confidence in (HIGH, MEDIUM) for m in matches)


def test_a_source_with_no_twin_comes_back_no_match(tmp_path):
    images = {"http://a/1.png": encode(make_image(4)),
              "http://b/9.png": encode(make_image(500))}
    source = {"S1": Listing("S1", "Orphan", ["http://a/1.png"])}
    target = {"T9": Listing("T9", "Unrelated", ["http://b/9.png"])}

    cache = HashCache(str(tmp_path / "c.json"))
    th, _ = hash_listings(target, cache, fetcher=make_fetcher(images, []))
    sh, _ = hash_listings(source, cache, fetcher=make_fetcher(images, []))

    matches = match_catalogues(source, target, sh, th)
    assert matches[0].confidence == NO_MATCH
    assert matches[0].target_key == ""


def test_results_are_sorted_by_tier():
    source = {k: Listing(k, k, []) for k in ("a", "b", "c")}
    target = {"t": Listing("t", "T", [])}
    matches = match_catalogues(
        source, target,
        {"a": [0b0], "b": [0xFFFFFFFFFFFFFFFF], "c": [0b0, 0b0]},
        {"t": [0b0]},
    )
    tiers = [m.confidence for m in matches]
    assert tiers == sorted(tiers, key=lambda t: {HIGH: 0, MEDIUM: 1,
                                                 LOW: 2, NO_MATCH: 3}[t])


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def test_mapping_csv_has_the_documented_columns(tmp_path):
    from match_by_photo import Match

    path = tmp_path / "out.csv"
    write_mapping([Match("S1", "Source", "T1", "Target", 3, 2, 4, HIGH)], str(path))

    rows = list(csv.reader(path.open(encoding="utf-8")))
    assert rows[0] == ["source_key", "source_title", "target_key", "target_title",
                       "best_distance", "votes", "photos_hashed", "confidence"]
    assert rows[1] == ["S1", "Source", "T1", "Target", "3", "2", "4", HIGH]


def test_no_match_rows_leave_distance_blank_rather_than_writing_minus_one(tmp_path):
    from match_by_photo import Match

    path = tmp_path / "out.csv"
    write_mapping([Match("S1", "Source", "", "", -1, 0, 2, NO_MATCH)], str(path))
    assert list(csv.reader(path.open(encoding="utf-8")))[1][4] == ""


def test_summary_reports_every_tier_and_the_cache_rate():
    from match_by_photo import Match

    cache = HashCache("")
    cache.hits, cache.misses = 3, 1
    text = summarise([Match("S1", "s", "T1", "t", 1, 2, 2, HIGH)], ["u"], cache)
    for tier in (HIGH, MEDIUM, LOW, NO_MATCH):
        assert tier in text
    assert "75%" in text
    assert "fetch failures:  1" in text


# ---------------------------------------------------------------------------
# load_catalogue dispatch
# ---------------------------------------------------------------------------

def test_csv_is_dispatched_to_the_vela_parser(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("TITLE,SKU,Photo 1\nDragon,S1,a.jpg\n", encoding="utf-8")
    assert list(load_catalogue(str(path))) == ["S1"]


def test_a_bom_prefixed_export_still_finds_its_title_column(tmp_path):
    """Etsy and Shopify both hand out BOM-prefixed CSVs."""
    path = tmp_path / "bom.csv"
    path.write_bytes("TITLE,SKU,Photo 1\nDragon,S1,a.jpg\n".encode("utf-8-sig"))
    assert list(load_catalogue(str(path))) == ["S1"]


def test_xlsx_is_dispatched_to_the_altera_parser(tmp_path):
    from openpyxl import Workbook

    path = tmp_path / "b.xlsx"
    book = Workbook()
    sheet = book.active
    sheet.title = "Products"
    sheet.append(["Handle", "Title", "Image Src"])
    sheet.append(["dragon", "Dragon", "a.jpg"])
    sheet.append(["dragon", "", "b.jpg"])
    book.save(path)

    listings = load_catalogue(str(path))
    assert listings["dragon"].images == ["a.jpg", "b.jpg"]


def test_an_unsupported_extension_says_what_it_expected(tmp_path):
    path = tmp_path / "c.pdf"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Vela/Etsy"):
        load_catalogue(str(path))
