echo "converting ..."
date

cd chunks

for file in *; do
  echo -n "${file}:  "
  sed  's/$/,/' -i $file
  sed  '1s/^/[\n/' -i $file
  sed  '$ s/,$/\n]/' -i $file 

  printf 'done\n'
done

echo "done"
date
