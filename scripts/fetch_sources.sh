#!/usr/bin/env bash
# Download the raw sources (not redistributed in this repo). Run from the repo root.
set -euo pipefail
UA="Mozilla/5.0"
mkdir -p data/raw/mappings data/raw/old data/raw/official
get() { curl -sSL -A "$UA" -o "$1" "$2"; }
# Gazette texts of the new laws (PRS India copies)
get data/raw/bns_prs.pdf  "https://prsindia.org/files/bills_acts/bills_parliament/2023/The%20Bharatiya%20Nyaya%20Sanhita,%202023.pdf"
get data/raw/bnss_prs.pdf "https://prsindia.org/files/bills_acts/bills_parliament/2023/The%20Bharatiya%20Nagarik%20Suraksha%20Sanhita,%202023.pdf"
get data/raw/bsa_prs.pdf  "https://prsindia.org/files/bills_acts/acts_parliament/2023/The%20Bharatiya%20Sakshya%20Bill,%202023.pdf"
# Repealed codes (India Code)
get data/raw/old/ipc.pdf  "https://www.indiacode.nic.in/bitstream/123456789/4219/1/THE-INDIAN-PENAL-CODE-1860.pdf"
get data/raw/old/crpc.pdf "https://www.indiacode.nic.in/bitstream/123456789/15272/1/the_code_of_criminal_procedure,_1973.pdf"
get data/raw/old/iea.pdf  "https://www.indiacode.nic.in/bitstream/123456789/15351/1/iea_1872.pdf"
# Public mapping tables
get data/raw/mappings/vakeel360.html      "https://vakeel360.com/ipc-to-bns"
get data/raw/mappings/vakeel360_crpc.html "https://vakeel360.com/crpc-to-bnss"
get data/raw/mappings/lawsikho.html       "https://lawsikho.com/blog/ipc-to-bns-conversion"
# Allahabad High Court BNS<->IPC comparison (87 MB); parsed pairs are committed in data/raw/official/
get data/raw/official/allahabad_newlaw24_vol2.pdf "https://allahabadhighcourt.in/ILR/newlaw24_vol2.pdf"
echo "done; now: python3 scripts/prepare_corpus.py && python3 scripts/prepare_old_codes.py && python3 scripts/build_benchmark.py"
