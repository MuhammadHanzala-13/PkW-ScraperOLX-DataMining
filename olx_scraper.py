"""
OLX Pakistan - Car Listings Scraper
Uses Microsoft Playwright (real Chromium browser) to bypass
Cloudflare bot protection that blocks simple HTTP requests.

Why Playwright?
  - Executes JavaScript (required by Cloudflare challenge pages)
  - Real TLS fingerprint (not flagged like Python requests)
  - Handles cookies & sessions automatically

Flow:
  1. Launch headless Chromium
  2. Visit OLX cars search page
  3. Extract __NEXT_DATA__ JSON from the live DOM
  4. Visit each listing detail page for full params
  5. Save raw CSV (no processing - just raw data)
"""

import json
import time
import random
import os
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
BASE_URL   = "https://www.olx.com.pk"
SEARCH_URL = "https://www.olx.com.pk/cars_c84"
OUTPUT_FILE = "data/olx_cars_raw.csv"

# ---------------------------------------------------------------------------
# UTILITY: Get __NEXT_DATA__ from a live browser page
# ---------------------------------------------------------------------------
def get_next_data(page, url, retries=2):
    """Navigate to URL in Playwright and extract __NEXT_DATA__ JSON."""
    for attempt in range(retries):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            # Wait for the script tag to appear
            page.wait_for_selector("#__NEXT_DATA__", timeout=15000)
            raw_json = page.eval_on_selector(
                "#__NEXT_DATA__", "el => el.textContent"
            )
            return json.loads(raw_json)
        except PlaywrightTimeout:
            print(f"  [TIMEOUT] Attempt {attempt+1}/{retries} for {url}")
            time.sleep(3)
        except Exception as e:
            print(f"  [ERROR] {e} at {url}")
            return None
    return None

# ---------------------------------------------------------------------------
# STEP 1: Extract listing summaries from search page __NEXT_DATA__
# ---------------------------------------------------------------------------
def extract_listings(page_data):
    """Pull car listing summaries from the search page JSON."""
    ads = []
    try:
        # Try primary path
        ads = (
            page_data
            .get("props", {})
            .get("pageProps", {})
            .get("ads", [])
        )
        # Try alternate path if empty
        if not ads:
            ads = (
                page_data
                .get("props", {})
                .get("pageProps", {})
                .get("listingProps", {})
                .get("ads", [])
            )
        # Try another alternate
        if not ads:
            ads = (
                page_data
                .get("props", {})
                .get("pageProps", {})
                .get("data", {})
                .get("ads", [])
            )
    except Exception as e:
        print(f"  [WARN] Could not extract ads list: {e}")
    return ads

# ---------------------------------------------------------------------------
# STEP 2: Extract detail params from listing page __NEXT_DATA__
# ---------------------------------------------------------------------------
def extract_detail(page_data):
    """Pull all parameters from a single car listing's page JSON."""
    params = {}
    try:
        ad = (
            page_data
            .get("props", {})
            .get("pageProps", {})
            .get("ad", {})
        )
        if not ad:
            ad = (
                page_data
                .get("props", {})
                .get("pageProps", {})
                .get("adData", {})
                .get("ad", {})
            )

        params["description"] = ad.get("description", "").replace("\n", " ").strip()
        params["posted_date"] = ad.get("createdAt") or ad.get("created_at")
        params["views"] = ad.get("viewsCount")

        # All structured params: Year, Mileage, Make, Fuel, etc.
        for p in ad.get("params", []):
            key   = p.get("key", "unknown")
            value = p.get("value", {})
            if isinstance(value, dict):
                params[key] = value.get("label") or value.get("key", "")
            else:
                params[key] = value
    except Exception as e:
        print(f"  [ERROR] Detail extraction: {e}")
    return params

# ---------------------------------------------------------------------------
# MAIN SCRAPER
# ---------------------------------------------------------------------------
def scrape(total_pages=5):
    """
    Main driver. Launches a real browser, scrapes N pages of OLX,
    fetches each listing detail, and saves everything raw to CSV.
    """
    os.makedirs("data", exist_ok=True)
    all_records = []

    print("=" * 60)
    print("  OLX Pakistan | Car Listings Scraper (Playwright)")
    print(f"  Target: {total_pages} pages (~{total_pages * 28} listings)")
    print("=" * 60)

    with sync_playwright() as pw:
        # Launch browser (headless=False lets you watch it work)
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="en-US",
        )
        page = context.new_page()

        for page_num in range(1, total_pages + 1):
            search_url = f"{SEARCH_URL}?page={page_num}"
            print(f"\n[PAGE {page_num}/{total_pages}] {search_url}")

            page_data = get_next_data(page, search_url)
            if not page_data:
                print(f"  [SKIP] No data on page {page_num}.")
                continue

            summaries = extract_listings(page_data)
            if not summaries:
                print(f"  [STOP] No listings found. Here are top-level keys:")
                print(f"         {list(page_data.get('props',{}).get('pageProps',{}).keys())}")
                break

            print(f"  Found {len(summaries)} listings. Fetching details...")

            for i, ad in enumerate(summaries):
                # --- Build base record from search summary
                record = {
                    "id":       ad.get("id"),
                    "title":    ad.get("title"),
                    "url":      ad.get("url"),
                    "price":    ad.get("price"),
                    "location": ad.get("location"),
                    "category": ad.get("category"),
                }

                listing_url = ad.get("url", "")
                if listing_url and not listing_url.startswith("http"):
                    listing_url = BASE_URL + listing_url

                if listing_url:
                    print(f"    ({i+1}/{len(summaries)}) {ad.get('title','')[:50]}...")
                    detail_data = get_next_data(page, listing_url)
                    if detail_data:
                        detail_params = extract_detail(detail_data)
                        record.update(detail_params)
                    # Polite random delay
                    time.sleep(random.uniform(1.5, 3.0))

                all_records.append(record)

            # Checkpoint save after every page
            pd.DataFrame(all_records).to_csv(OUTPUT_FILE, index=False)
            print(f"  [CHECKPOINT] {len(all_records)} total records saved.")

        browser.close()

    print("\n" + "=" * 60)
    print(f"  Done. {len(all_records)} records saved to {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    # Adjust total_pages for more data (each page ≈ 28 listings)
    scrape(total_pages=5)
