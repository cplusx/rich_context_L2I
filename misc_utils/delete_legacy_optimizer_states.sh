EXPT_NAME=$1
cd experiments/$EXPT_NAME

# Get a list of files matching the pattern, sort them numerically, and skip the last two files
dirs_to_delete=$(find . -maxdepth 1 -type d -name 'epoch=*' | sort -V | head -n -2)

# Loop through and delete the directories
for dir in $dirs_to_delete; do
    echo "Deleting $dir"
    rm -r "$dir"
done