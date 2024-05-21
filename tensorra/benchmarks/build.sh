# Install Monetdb

# Build database
monetdbd create db
# Use an idle port
monetdbd set port=23332 db
monetdbd start db
monetdb create bench
monetdb release bench
# Passwordless start
echo "user=monetdb" > .monetdb
echo "password=monetdb" >> .monetdb
# mclient -d bench
