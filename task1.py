from elasticsearch import Elasticsearch

es = Elasticsearch(HOST="http://localhost", PORT=9200)
es = Elasticsearch()

print(es.indices.delete(index = "first_index"))

