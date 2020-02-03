"""Set up KP registry database."""
import os
import sqlite3

import yaml

# get edges provided by KPs
with open('examples/kps2.yml', 'r') as f:
    pedges = yaml.load(f, Loader=yaml.SafeLoader)

# remove db if it exists
db_filename = 'kps.db'
if os.path.exists(db_filename):
    os.remove(db_filename)

conn = sqlite3.connect(db_filename)

c = conn.cursor()

# Create table
c.execute('''CREATE TABLE knowledge_providers
             (url text UNIQUE, source_type text, edge_type text, target_type text)''')

values = [
    (url, details['source_type'], details['edge_type'], details['target_type'])
    for url, details in pedges.items()
]
# Insert rows of data
c.executemany("INSERT INTO knowledge_providers VALUES (?, ?, ?, ?)", values)

# Save (commit) the changes
conn.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()
