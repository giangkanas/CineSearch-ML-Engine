from elasticsearch import Elasticsearch

es = Elasticsearch(HOST="http://localhost", PORT=9200)
   
print("ΑΝΑΖΗΤΗΣΗ:")
Input = str(input())

# Query με το οποίο θα κάνω αναζήτηση με βάση το αλφαριθμητικό που δίνω ως input 
body = {
    "from" : 0,
    "size" : 9000,
    "query" : {
        "match" : {
            "title" : Input
            }
        }
    }

res = es.search(index="movies" , body=body)     #Με την εντολή αυτή εκτελώ αναζήτηση στο index "movies"
                                                #με βάση το query που δημιούργησα ("body")
                                                
all_hits = res["hits"]["hits"]                  #Αφού κοίταξα το περιεχόμενο του dictionary res είδα ότι 
                                                #το κύριο περιεχόμενο το οποίο αναζητώ βρίσκεται στο υπο-λεξικό
                                                #res["hits"]["hits"]     

# Εκτύπωση των αποτελεσμάτων που επιστράφηκαν από την Elasticsearch
for i in all_hits:
    print(i["_source"]["title"],i["_source"]["genres"])



 

    
   

   

