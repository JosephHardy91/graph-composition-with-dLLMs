FROM pgvector/pgvector:pg16

# Copy schema_and_cte.sql and the entrypoint initialization wrapper
COPY starter/schema_and_cte.sql /opt/sql/schema_and_cte.sql
COPY docker/init-db.sh /docker-entrypoint-initdb.d/01-init-db.sh

RUN chmod +x /docker-entrypoint-initdb.d/01-init-db.sh
