$Language = "en"

$WikiDumpName = "${Language}wiki-latest-pages-articles.xml.bz2"
$WikiDumpDownloadUrl = "https://dumps.wikimedia.org/${Language}wiki/latest/$WikiDumpName"

Write-Host "Downloading the latest $Language-language Wikipedia dump from $WikiDumpDownloadUrl..."
Invoke-WebRequest -Uri $WikiDumpDownloadUrl -OutFile $WikiDumpName -UseBasicP

Write-Host "Successfully downloaded the latest $Language-language Wikipedia dump to $WikiDumpName"
