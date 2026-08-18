# Vietnamese spell-correction data

`vietnamese_lexicon.json` is generated from the Viet74K word and phrase list by
Ho Ngoc Duc, redistributed by the `duyet/vietnamese-wordlist` project under
GPL-2.0.

- Source: https://github.com/duyet/vietnamese-wordlist
- Input file: `Viet74K.txt`
- License: GPL-2.0

To rebuild the JSON from an acquired source file:

```powershell
python scripts/build_vietnamese_lexicon.py path\to\Viet74K.txt
```

The builder accepts only NFC-normalized one- and two-token alphabetic entries.
It derives accentless candidate groups from the accepted tokens and uses
two-token source entries as conservative bigram evidence.
