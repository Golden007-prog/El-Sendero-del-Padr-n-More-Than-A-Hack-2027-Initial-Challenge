# Match inputs

The competition match is supplied to authorized participants by the organizers. Obtain and unpack the bundle through the access instructions sent with enrollment. This repository does not provide or redistribute that match.

TRACE uses these two automatic tactical-tracking files:

```text
<authorized-bundle-root>/
└── tactical/
    ├── BAR-ATH_tactical_atd.xml
    └── BAR-ATH_tactical_atd.txt
```

Pass the parent bundle directory to the inference entry point:

```powershell
.\.venv\Scripts\python.exe -I -B infer_candidate.py `
  --data-dir "<authorized-bundle-root>" `
  --output "outputs/El Sendero del Padrón.csv"
```

The path should point above `tactical/`, not at either file. Preserve both filenames. Video, Smart Tagging, homography and corrected tracking are not required for this inference route. No possession reference file is requested.

The inherited organizer helper `games.py` also supports the `MTAH_DATA_DIR` environment variable or a local `data/` directory for its general asset lookup. The command above supplies the location explicitly with `--data-dir`.

Keep match assets outside the public repository. Local `data/`, `inputs/`, `outputs/`, generated CSVs and common match/cache formats are ignored by Git. Input access does not grant redistribution rights; consult the organizers’ [data documentation](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md) and [rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md).

The external Sportec training dataset is separate from the challenge match. Its versioned references and attribution are in [sportec-required-assets.json](sportec-required-assets.json), [the model card](docs/MODEL_CARD.md) and [NOTICE.md](NOTICE.md). It is not needed to run the bundled fitted models.
