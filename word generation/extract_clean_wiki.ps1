# .\extract_clean_wiki.ps1 -WIKI_DUMP_FILE_IN "enwiki-latest-pages-articles.xml.bz2"
#Set-StrictMode -Version Latest

param (
    [string]$WIKI_DUMP_FILE_IN
)

if (-not $WIKI_DUMP_FILE_IN) {
    Write-Host "Error: WIKI_DUMP_FILE_IN parameter is required."
    exit 1
}

$WIKI_DUMP_FILE_OUT = [System.IO.Path]::ChangeExtension($WIKI_DUMP_FILE_IN, ".txt")

# Clone the WikiExtractor repository if it doesn't exist
if (-not (Test-Path "wikiextractor")) {
    git clone https://github.com/attardi/wikiextractor.git
}
Set-Location -Path "wikiextractor"
Write-Host "Extracting and cleaning $WIKI_DUMP_FILE_IN to $WIKI_DUMP_FILE_OUT..."

# Run the WikiExtractor and process the output
python -m wikiextractor.WikiExtractor $WIKI_DUMP_FILE_IN --processes 8 -q -o - |
    Where-Object { $_ -notmatch '^\s*$' -and $_ -notmatch '^<doc id=' -and $_ -notmatch '</doc>$' } |
    Set-Content -Path $WIKI_DUMP_FILE_OUT

Write-Host "Successfully extracted and cleaned $WIKI_DUMP_FILE_IN to $WIKI_DUMP_FILE_OUT"