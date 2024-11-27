#!/bin/bash

# Configuration-Engine setup:
git submodule init
git submodule update

pip install -r requirements.txt

# AMiner installation:
cd $HOME
wget https://raw.githubusercontent.com/ait-aecid/logdata-anomaly-miner/main/scripts/aminer_install.sh

FILE="aminer_install.sh"
VAR_NAME="BRANCH"
NEW_VALUE="development" # change to development version is necessary

chmod +x $FILE
sed -i.bak "s/^$VAR_NAME=.*/$VAR_NAME=\"$NEW_VALUE\"/" "$FILE"
rm "$FILE.1" "$FILE.bak"
echo "Updated $VAR_NAME to \"$NEW_VALUE\" in $FILE"

./aminer_install.sh

rm aminer_install.sh

# link AMiner parsers:
SRC_DIR="/etc/aminer/conf-available/ait-lds"
DEST_DIR="/etc/aminer/conf-enabled"

for file in "$SRC_DIR"/*; do
    filename=$(basename "$file")
    # create a symbolic link in the destination directory
    if [ ! -e "$DEST_DIR/$filename" ]; then
        echo "Linking $file to $DEST_DIR/$filename"
        sudo ln -s "$file" "$DEST_DIR/$filename"
    else
        echo "Skipping $file, link already exists at $DEST_DIR/$filename"
    fi
done