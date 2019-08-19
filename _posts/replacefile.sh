#!/bin/bash
cd "$(dirname "$0")"
filename=$1

echo $1

sed -i 's/ \$\$/ \{% math %\}/g' $filename
sed -i 's/^\$\$/\{% math %\}/g' $filename
sed -i 's/\$\$/\{% endmath %\} /g' $filename

sed -i 's/[ \n]\$/ \{% m %\}/g' $filename
sed -i 's/\$ /\{% em %\} /g' $filename
sed -i 's/\$,/\{% em %\},/g' $filename
sed -i 's/\$\./\{% em %\}\./g' $filename

