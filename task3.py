from elasticsearch import Elasticsearch
import pandas as pd
import math
from sklearn.cluster import KMeans
import numpy as np

np.random.seed(1053577)

df_ratings = pd.read_csv('ratings.csv')
df1_movies = pd.read_csv('movies.csv')
df_ratings = df_ratings.drop("timestamp",axis=1)

# ============================ ΒΑΘΜΟΛΟΓΙΑ ΠΟΥ ΕΧΕΙ ΒΑΛΕΙ Ο ΧΡΗΣΤΗΣ =================================

    # =================================== ΥΠΟΛΟΓΙΣΜΟΣ ΜΕΣΟΥ ΟΡΟΥ ΚΑΘΕ ΕΙΔΟΥΣ ΓΙΑ ΚΑΘΕ ΧΡΗΣΤΗ  ==========================================


df1_movies.genres = df1_movies.genres.str.split('|')    #Κάνω str.split() στη στήλη genres
df2_genre = df1_movies.explode("genres")                #Κάνω explode στο dataframe με βάση την στήλη genres
                                                        #και για κάθε διαφορετικό genre δημιουργείται καινούρια γραμμή στο dataframe
df2_genre = df2_genre.drop("title",axis = 1)            #Κρατάω μόνο τα ζευγάρια movieId, genre

df3 = pd.merge(left = df_ratings, how="right" , right = df2_genre,      #Κάνω merge το dataframe των ratings με το df2_genre με σκοπό
               left_on="movieId" , right_on="movieId")                  #να έχω τις βαθμολογίες και τα είδη στο ίδιο dataframe
                                                                        #με το how = "right" διασφαλίζω ότι στο τελικό dataframe df3 θα 
                                                                        #περιλαμβάνονται και οι ταινίες οι οποίες δεν έχουν βαθμολογηθεί
                                                                        #από κανένα χρήστη 

df4 = df3.groupby(["userId","genres"],as_index=False)["rating"].mean()  #Ομαδοποιώ τις γραμμές του df3 με βάση τα userId, genres και στην
                                                                        #συνέχεια βρίσκω τον μέσο όρο της βαθμολογίας κάθε ζεύγους 
                                                                        #userId - genres    
df4 = df4.pivot(index="userId", columns="genres", values="rating")      #Τέλος για να μην επαναλαμβάνονται στοιχεία σε κάθε γραμμή 
                                                                        #(τα userId και τα genres) αλλάζω την δομή του dataframe 
                                                                        #με την βοήθεια της pivot() σχηματίζοντας ένα νέο με πλήθος
                                                                        #γραμμων ίσο με το πλήρος των χρηστών και με πληθος στηλών ίσο 
                                                                        #με το πλήθος των genres   

df5_genreMean = df3.groupby("genres",as_index=False)["rating"].mean()   #Βρίσκω τη μέση βαθμολογία κάθε είδους προκειμένου να γεμίσω
                                                                        #τα NaN κάθε είδους του df4   

genresDict = {df5_genreMean.genres.iloc[i] : df5_genreMean.rating.iloc[i] #Σχηματίζω ένα λεξικο με τα ζευγάρια key-value
              for i in range(len(df5_genreMean))}                         #να είναι τα ζευγάρια genre-rating

df4 = df4.fillna(genresDict)                                              #Γεμίζω τα NaN με βάση αυτό το λεξικό  


    #======================================= ΣΥΣΤΑΔΟΠΟΙΗΣΗ ΤΩΝ ΧΡΗΣΤΩΝ ========================================================


kmeans = KMeans(n_clusters=8)           #Θεωρώ τον αριθμό των συστάδων ίσο με 8
kmeans.fit(df4)                         #Εκπαιδεύω το μοντέλο με βάση το σύνολο δεδομένων που προέκυψε από πάνω  
prediction = kmeans.predict(df4)        #Γίνεται η συσταδοποίηση των χρηστών με βάση το df4

prediction = pd.Series(prediction,index=[i for i in range(1,len(prediction)+1)]) #Εδώ μετατρέπω το prediction σε Series
                                                                                 #αλλάζοντας όμως το index έτσι ώστε να το
                                                                                 #προσθέσω στο df4 (το df4 ξεκινάει με index=1)   
df4["cluster"] = prediction             #Προσθήκη του prediction στο σύνολο δεδομένων


    #======================================= ΣΥΜΠΛΗΡΩΣΗ ΒΑΘΜΟΛΟΓΙΩΝ ΠΟΥ ΛΕΙΠΟΥΝ ================================================

df6 = pd.merge(left = df_ratings, right = df4 ,             #Εφόσον θέλω να βρω την μέση βαθμολογία κάθε ταινίας για κάθε
               left_on="userId" , right_on = df4.index)     #συστάδα. Έτσι χρειάζεται να προσθέσω στον αρχικό πίνακα ratings την στήλη
df6 = df6[["userId","movieId","rating","cluster"]]          #clusters. Κρατάω μόνο τα δεδομένα που χρειάζομαι.
                                                        
df7_ratingByCluster = df6.groupby(["cluster","movieId"] , as_index=False)["rating"].mean()      #Βρίσκω την μέση βαθμολογία κάθε ταινίας
                                                                                                #για κάθε cluster
df7_ratingByCluster = df7_ratingByCluster.pivot(index="movieId",columns="cluster",values="rating") #Φέρνω τον πίνακα στην μορφή 
                                                                                                   #movieId x cluster


df8 = df_ratings.pivot(index="userId" , columns="movieId", values="rating")    #Φέρνω τον πίνακα στην μορφή movieId x cluster
                                                                                                   


for i in range(len(df8)):                   #Για κάθε γραμμή του df8 δηλαδή για κάθε χρήστη
    user_cluster = df4["cluster"].iloc[i]   #cluster στο οποίο ανήκει ο χρήστης i
    user_cluster_ratings = df7_ratingByCluster[user_cluster] #τα ratings των ταινιών για τον user i
    df8.iloc[i].fillna(user_cluster_ratings, inplace=True)   #αντικατάσταση των NaN για τον χρήστη i
    # print(df.iloc[i])
    
# ===========================================================================================================   



#============================= ΜΕΤΡΙΚΗ ΟΜΟΙΟΤΗΤΑΣ ELASTICSEARCH ===================================

es = Elasticsearch(HOST="http://localhost", PORT=9200)

df = pd.read_csv('ratings.csv') #Βάζω το ratings.csv σε ένα dataframe

        
print("ΑΝΑΖΗΤΗΣΗ:")
Input = str(input())
print("USER_ID:")
user_id = int(input())

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


df10 = pd.DataFrame.from_dict([i['_source'] for i in match]) #Μετατροπή του λεξικού match σε Dataframe

df10["BM25rating"]=[i["_score"] for i in match]   #Προσθέτω τη στήλη order_rating στο οποίο βάζω score της μετρικής 
                                                   #ομοιότητας της Elasticsearch


#============================= ΜΕΣΟΣ ΟΡΟΣ ΒΑΘΜΟΛΟΓΙΩΝ ΚΑΘΕ ΤΑΙΝΙΑΣ ===================================

df11_MeanRating = df_ratings.groupby("movieId")["rating"].mean().to_frame()  #Με την συνάρτηση groupby ομαδοποιώ τις γραμμές του dataframe df
                                                                    #με βάση το movieId και από κάθε ομάδα με την mean() βρίσκω την μέση τιμή
                                                                    #του πεδίου "rating". Τέλος το to_frame μετατρέπει το Series σε dataframe
                                                                    #Βρίσκει τον μέσο όρο κάθε βαθμολογημένης ταινίας

df11_MeanRating.rename(columns = {"rating" : "meanRating"},inplace=True)

df12 = pd.merge(left = df10 , how="left" , right = df11_MeanRating,
               left_on="movieId" , right_on="movieId")  #Κάνω merge τα dataframes df10, df11_MeanRating
                                                        #και το dataframe που προκύπτει περιλαμβάνει
                                                        #τα στοιχεία κάθε ταινίας df10 και ουσιαστικά
                                                        #προσθέτει τον Μέσο Όρο που υπάρχει στο df11_MeanRating 
                                                        #με το how = "left" διασφαλίζω ότι στο τελικό dataframe df12 θα περιλαμβάνονται και οι ταινίες
                                                        #οι οποίες δεν έχουν βαθμολογηθεί από κανένα χρήστη 

#============================= ΒΑΘΜΟΛΟΓΙΕΣ ΧΡΗΣΤΗ ΓΙΑ ΚΑΘΕ ΤΑΙΝΙΑ ===================================

df13 = pd.DataFrame(df8.loc[user_id])       #Από το df8 παίρνω την γραμμή του user_id που εχει τις βαθμολογιες
                                            #για ολες τις βαθμολογημενες ταινιες
df13.rename(columns = {user_id : "userRating"}, inplace = True)

df14 = pd.merge(left = df12 , right = df13 , how = "left",
               left_on="movieId" , right_on="movieId")  #Στο dataframe df12 προσθέτω ακόμα
                                                        #ένα πεδίο για κάθε γραμμή που είναι η βαθμολογία του χρήστη
                                                        #Για τον ίδιο λόγο με πριν έχω βάλει το how = left για να διασφαλίσω ότι στο df14
                                                        #θα περιλαμβάνονται και οι ταινίες οι οποίες δεν έχουν βαθμολογηθεί από τον ζητούμενο χρήστη                              

#============================= ΜΕΤΡΙΚΗ ΠΟΥ ΠΕΡΙΛΑΜΒΑΝΕΙ ΟΛΑ ΤΑ ΠΑΡΑΠΑΝΩ ===================================
def rating_function(vec):
    orderRating=vec[0]
    meanRating=vec[1]
    userRating=vec[2]
    
    if(math.isnan(userRating) == True and math.isnan(meanRating) == False):     
        return meanRating + orderRating
    elif(math.isnan(userRating) == True and math.isnan(meanRating) == True):
        return orderRating
    else:
        return meanRating + userRating + orderRating

df15 = df14;  
df15["metric"] = df15[["BM25rating","meanRating","userRating"]].apply(rating_function,axis=1) #Εφαρμόζω την συνάρτηση rating_function για κάθε
                                                                                            #γραμμή του df15 (axis = 1) περνώντας σαν όρισμα ένα vector
                                                                                            #που περιλαμβάνει τα δεδομένα με τα οποία θα υπολογιστεί
                                                                                            #η νέα μετρική

df16_final = df15.sort_values(["metric"],ascending=False) #Ταξινομώ το df15 με βάση τo metric κατά φθίνουσα σειρά

print(df16_final[["title","genres","metric"]])       #Εκτύπωση αποτελέσματος


# for i in range(len(df7_final)):
#     print(df7_final.iloc[i].title,df7_final.iloc[i].genres,df7_final.iloc[i].metric)



   




