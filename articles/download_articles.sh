#!/bin/bash

# it automatically downloads the articles from scihub by using https://github.com/dougy147/scitopdf
# usage: ./download_articles.sh articles_db.txt

articles="$1"

[ -z "$articles" ] && echo "You forgot the list of articles as argument." && exit 1



tempfile=$(mktemp "/tmp/articles-doi.XXXXXX")

sed '1d;2d' < "$articles" | awk -F',' '{print $2}' | awk '{$1=$1};1' > "$tempfile"
scitopdf -D . -p -l "$tempfile"

rm "$tempfile"
exit 0
