The three supplied source datasets. **Read-only** — nothing in this repository
writes here, and altering them is prohibited outright (CON-3, brief §9 rule 3).

```
AFR/          86 JSONL files, one per month, 780 MB, 219,538 records.
              Fields: HEADLINE, SUBHEAD, INTRO, TEXT, NEWSPAPER, PUBLICATIONDATE.
              AFR_20150201-20150201.jsonl is empty; most files end with a blank line.
ASX/          18 JSONL files named by company. Fields: ticker (with .AX suffix),
              date, open, high, low, close, volume. 1,774 rows each.
"RBA Rates"/  RBA-rates.csv and RBA-rates.jsonl, 175 decisions. Use the CSV — the
              JSONL carries a UTF-8 BOM that corrupts its first key.
```

Ingested artifacts are built from these into `DATA_DIR` (`./data` by default) by
`python -m src.ingest`, which is the only code that reads this directory.
Override the location with `SOURCE_DATA_DIR`.
