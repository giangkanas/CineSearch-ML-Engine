from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import numpy as np
from zkeras.models import Sequential
from keras.layers import Dense 
from elasticsearch import Elasticsearch
import math

es = Elasticsearch(HOST="http://localhost", PORT=9200)

np.random.seed(1053577)


df1_ratings = pd.read_csv('ratings.csv') #Βάζω το ratings.csv σε ένα dataframe

        
print("ΑΝΑΖΗΤΗΣΗ:")
Input = str(input())
print("USER_ID:")
user_id = int(input())

# Input = "toy story"
# user_id = 5

#============================= ΜΕΤΡΙΚΗ ΟΜΟΙΟΤΗΤΑΣ ELASTICSEARCH ===================================

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

match = res["hits"]["hits"]                     #Αφού κοίταξα το περιεχόμενο του dictionary res είδα ότι 
                                                #το κύριο περιεχόμενο το οποίο αναζητώ βρίσκεται στο υπο-λεξικό
                                                #res["hits"]["hits"]


df1 = pd.DataFrame.from_dict([i['_source'] for i in match]) #Μετατροπή του λεξικού match σε Dataframe

df1["BM25rating"]=[i["_score"] for i in match]   #Προσθέτω τη στήλη order_rating στο οποίο βάζω score της μετρικής 
                                                   #ομοιότητας της Elasticsearch
                                                    
#============================= ΜΕΣΟΣ ΟΡΟΣ ΒΑΘΜΟΛΟΓΙΩΝ ΚΑΘΕ ΤΑΙΝΙΑΣ ===================================

df2_MeanRating = df1_ratings.groupby("movieId")["rating"].mean().to_frame()  #Με την συνάρτηση groupby ομαδοποιώ τις γραμμές του dataframe df
                                                                    #με βάση το movieId και από κάθε ομάδα με την mean() βρίσκω την μέση τιμή
                                                                    #του πεδίου "rating". Τέλος το to_frame μετατρέπει το Series σε dataframe
                                                                    #Βρίσκει τον μέσο όρο κάθε βαθμολογημένης ταινίας

df2_MeanRating.rename(columns = {"rating" : "meanRating"},inplace=True)

df3 = pd.merge(left = df1 , how="left" , right = df2_MeanRating,
               left_on="movieId" , right_on="movieId")  #Κάνω merge τα dataframes df1, df2_MeanRating
                                                        #και το dataframe που προκύπτει περιλαμβάνει
                                                        #τα στοιχεία κάθε ταινίας df1 και ουσιαστικά
                                                        #προσθέτει τον Μέσο Όρο που υπάρχει στο df2_MeanRating 
                                                        #με το how = "left" διασφαλίζω ότι στο τελικό dataframe df3 θα περιλαμβάνονται και οι ταινίες
                                                        #οι οποίες δεν έχουν βαθμολογηθεί από κανένα χρήστη 




df4_movies = pd.read_csv('movies.csv')

titles = df4_movies["title"]
genres = df4_movies["genres"]



#============================= ΒΑΘΜΟΛΟΓΙΕΣ ΧΡΗΣΤΗ ΓΙΑ ΚΑΘΕ ΤΑΙΝΙΑ ===================================

#==================== ΤΙΤΛΟΙ ΩΣ WORD EMBEDDINGS ΚΑΙ GENRES ΩΣ ONE HOT ENCODING =====================

#Με το Doc2Vec μπορώ να παράξω ένα διάνυσμα από ένα document. Document θεωρείται οποιαδήποτε αλληλουχία λέξεων =>κάθε τίτλος είναι
#ένα document

# titles = [title.lower() for title in titles]
# titles = [TaggedDocument(words = doc.split(), tags = [str(i)])        #Το taggeddocument είναι μία δομή που χρησιμοποιείται με το
#           for i, doc in enumerate(titles)]                          #doc2vec και περιλαμβάνει τις λέξεις του document σε μορφή 
#                                                                     #λίστας και το index του document
# titles = [TaggedDocument(words = nltk.word_tokenize(doc.lower()), tags = [str(i)]) for i, doc in enumerate(titles)]
# model = Doc2Vec(titles, min_count=1,vector_size=200, epochs = 30)   #Δημιουργούμε το μοντέλο εκπαιδεύοντάς το με την λίστα από 
#                                                                     #taggedDocuments που δημιουργήσαμε, θέσαμε τον αρίθμο των epochs
#                                                                     #(πόσες φορές θα διαπεραστούν όλα τα documents) ίσο με 30
#                                                                     #έτσι ώστε να αυξήσουμε την ακρίβεια. Τέλος το διάνυσμα που 
#                                                                     #θα πάρουμε για κάθε τίτλο έχει μέγεθος 200.            
# model.save("titles_doc2vec.model")        

              

model = Doc2Vec.load("titles_doc2vec.model")
titlesVectors = [model.docvecs[str(i)] for i in range(len(titles))]         #WORD EMBEDDING τίτλων

#========================================== GENRES ΩΣ ONE HOT ENCODING ===============================================

genre_labels = pd.Series(i.split("|") for i in genres)                 #Χωρίζω τα είδη 
genre_labels = genre_labels.explode()                                  #Βάζω κάθε είδος σε διαφορετική γραμμή    
genre_labels = genre_labels.rename("genres")
genre_labels = genre_labels.groupby(genre_labels).size()     #Ομαδοποιώ ανά είδος (με το groupby πρέπει να χρησιμοποιήσω μια συναρτηση
                                                              #εδω χησιμοποιώ την size() δεν επηρεάζεται το τελικό αποτέλεσμα από αυτό

genre_labels = pd.DataFrame(genre_labels.index)                 #Εδώ έχω όλα τα διαθέσιμα είδη που υπάρχουν      

genre_labels_OneHot = pd.get_dummies(genre_labels.genres)          

genres = pd.Series(i.split("|") for i in genres)
genreVectors = []
for i in genres:
    onehot = [0 for i in range(len(genre_labels))]                 #Αρχικοποιώ την oneHot μορφη για κάθε διαφορετική ταινία
    for j in i:
        genre_id = genre_labels[(genre_labels ['genres'] == str(j))].index          #Id του genre j
        onehot2 = genre_labels_OneHot.iloc[genre_id].values.tolist()[0]             #Η αναπαράσταση one hot του genre j
        for k in range(len(genre_labels)):
            onehot[k] = onehot[k] + onehot2[k]                         #Προσθέτω για κάθε κατηγορία της ταινίας το onehot
    genreVectors.append(onehot)                                 


# ================================ ΔΙΑΝΥΣΜΑΤΑ ΕΙΣΟΔΟΥ - ΕΞΟΔΟΥ ΝΕΥΡΩΝΙΚΟΥ =================================================
InputVector=[]
for i in range(len(titlesVectors)):
    InputVector.append(np.concatenate((titlesVectors[i],genreVectors[i])))      #Για κάθε ταινια δημιουργώ διάνυσμα μεγέθους 220
                                                                                #200 απο word embedding και 20 απο one hot        
df4_movies["vector"] = InputVector

df5 = pd.merge(left = df1_ratings , right = df4_movies , left_on="movieId" , right_on="movieId")
df5 = df5[["userId","movieId","rating","vector"]]

df6_ratedMovies = df5[df5["userId"]==user_id]    #Βρίσκω τις ταινίες που έχει βαθμολογήσει ο user_id που δόθηκε σαν είσοδος
df7_NotRatedMovies = df4_movies[~df4_movies["movieId"].isin(df6_ratedMovies["movieId"].tolist())]   #Βρίσκω τις ταινίες που ΔΕΝ έχει βαθμολογήσει ο 
                                                                                                    #user_id για το testSet                                             

x_train = df6_ratedMovies["vector"]
x_train = np.matrix(x_train.tolist())       #Διάνυσμα εισόδου για εκπαίδευση νευρωνικού 

x_test = df7_NotRatedMovies["vector"]
x_test = np.matrix(x_test.tolist())         #Διάνυσμα εισόδου για test νευρωνικού 


y_train = df6_ratedMovies["rating"]    #Διάνυσμα με τις βαθμολογίες που έχει δώσει ο user_id βαθμολογίες (0.5 εως 5)
y_train = [i for i in y_train]         #Με το νευρωνικό που θα φτιάξω θα βρίσκω την πιθανότητα για κάθε μία απο τις πιθανές βαθμολογίες
                                       #οπότε πρέπει να μετατρέψω το y train σε one hot encoding 


realRatings = []
for i in range(5 , 51 , 5):
    realRatings.append(i/10)
realRatings = pd.Series(realRatings)        #Βάζω σε μία λίστα όλα τα πιθανά ratings
ratingsDict = realRatings.to_dict()    #Γία να φτιάξω το dictionary παίρνω τα ζευγη Key - value 
                                                                            #realRatings.iloc[i] : i . Αυτό το dictionary θα χρησιμοποιηθεί στην
                                                                            #μετατροπή των εξόδων του νευρωνικού από κλάσεις σε πραγματικές
                                                                            #βαθμολογίες
# ratingsDict = realRatings.to_dict()
onehot_encoded = list()
for i in y_train:
    dig = []
    for index,j in enumerate(realRatings):
        if i!=j: dig.append(0)
        else: dig.append(1)
    onehot_encoded.append(dig)

y_train = np.matrix(onehot_encoded)  
    
# ============================================== ΝΕΥΡΩΝΙΚΟ ΔΙΚΤΥΟ =================================================================

neural = Sequential([                                                       #Το νευρωνικό δικτυο που έφτιαξα έχει 2 κρυφά επίπεδα 
    Dense(units = 32, input_dim=220, activation="relu"),                    #(από 32 νευρωνες) και ένα επίπεδο εξόδου με 10 νευρώνες (πιθανότητα           
    Dense(units = 32, activation="relu"),                                   #κάθε κλάσης) 
    Dense(units = 10, activation="softmax")])

neural.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"] )

neural.fit(x_train,y_train,epochs=300)                  #Εκπαιδεύω το νευρωνικό μου με x_train , y_train

y_test = neural.predict_classes(x_test)                        #Προβλέπω τις κλάσεις που αντιστοιχούν στο x_test (στις ταινιες που δεν εχει
                                                               #βαθμολογήσει)
y_test = [ratingsDict[i] for i in y_test]      #Μετατρέπω τις κλάσεις σε πραγματικές βαθμολογίες


df7_NotRatedMovies.insert(4,"rating",y_test,True)      #Βάζω στην θέση 4 καινούρια στήλη με όνομα rating και 
                                                        #δεδομένα τα y_test δηλαδη τις προβλέψεις του νευρωνικου


df6_ratedMovies = df6_ratedMovies[["movieId","rating"]]
df7_NotRatedMovies = df7_NotRatedMovies[["movieId","rating"]]

# user_rating = user.append(not_user)

df8_allMovies = df6_ratedMovies.append(df7_NotRatedMovies)   #Εδω έχω καταφέρει να φτίαξω ένα dataframe που
df8_allMovies = df8_allMovies.rename(columns = {"rating" : "userRating"})       #έχει βαθμολογία για κάθε ταινία από τον χρήστη

#============================= ΒΑΘΜΟΛΟΓΙΕΣ ΧΡΗΣΤΗ ΓΙΑ ΚΑΘΕ ΤΑΙΝΙΑ ===================================

df9 = pd.merge(left = df3 , right = df8_allMovies ,         
               left_on="movieId" , right_on="movieId")

#============================= ΜΕΤΡΙΚΗ ΠΟΥ ΠΕΡΙΛΑΜΒΑΝΕΙ ΟΛΑ ΤΑ ΠΑΡΑΠΑΝΩ ===================================
def rating_function(vec):
    orderRating=vec[0]
    meanRating=vec[1]
    userRating=vec[2]
    
    if math.isnan(meanRating):
        return orderRating + userRating
    else:
        return meanRating + userRating + orderRating
    
df10 = df9;  
df10["metric"] = df10[["BM25rating","meanRating","userRating"]].apply(rating_function,axis=1) #Εφαρμόζω την συνάρτηση rating_function για κάθε
                                                                                            #γραμμή του df10 (axis = 1) περνώντας σαν όρισμα ένα vector
                                                                                            #που περιλαμβάνει τα δεδομένα με τα οποία θα υπολογιστεί
                                                                                            #η νέα μετρική

df11_final = df10.sort_values(["metric"],ascending=False) #Ταξινομώ το df10 με βάση τo metric κατά φθίνουσα σειρά

print(df11_final[["title","genres","metric"]])       #Εκτύπωση αποτελέσματος



















