from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pandas as pd

es = Elasticsearch(HOST="http://localhost", PORT=9200)

if es.indices.exists(index="movies"):
    es.indices.delete(index="movies")     #ΔΙΑΓΡΑΦΕΙ ΤΟ index ΠΟΥ ΥΠΑΡΧΕΙ ΣΤΗΝ elasticsearch ΜΕ ΣΚΟΠΟ ΝΑ ΤΟ ΞΑΝΑΦΤΙΑΞΩ

es.indices.create(index = "movies")

df = pd.read_csv('movies.csv')          #Βάζω το movies.csv σε ένα dataframe
df1 = df.to_dict(orient='records')      #Μετατρέπω το dataframe στο οποίο αποθήκευσα το csv file σε λίστα με dictionaries
                                        #γιατί το δεύτερο όρισμα της συνάρτησης bulk το απαιτεί
                                        #records : list like [{column -> value}, … , {column -> value}]

bulk(es, df1, index='movies',doc_type='movie', raise_on_error=True) #Με το bulk φορτώνω στο instance της elasticsearch ('es'),
                                                                    #στο index που δημιούργησα ("movies"), το dictionary που δημιούργησα         





 

    
   

   

