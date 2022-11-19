from elasticsearch import Elasticsearch
import pandas as pd
import math

es = Elasticsearch(HOST="http://localhost", PORT=9200)

df = pd.read_csv('ratings.csv') #Βάζω το ratings.csv σε ένα dataframe

        
print("ΑΝΑΖΗΤΗΣΗ:")
Input = str(input())
print("USER_ID:")
user_id = int(input())

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

df2_MeanRating = df.groupby("movieId")["rating"].mean().to_frame()  #Με την συνάρτηση groupby ομαδοποιώ τις γραμμές του dataframe df
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



#============================= ΒΑΘΜΟΛΟΓΙΕΣ ΧΡΗΣΤΗ ΓΙΑ ΚΑΘΕ ΤΑΙΝΙΑ ===================================
df4_user_ratings = df[df['userId'] == user_id]  #Από το dataframe με τα ratings df κρατάω μόνο τις γραμμές που έχουν userId αυτό που δόθηκε
                                                #σαν είσοδο 
                                            
df5 = pd.merge(left = df3 , right = df4_user_ratings , how = "left",
               left_on="movieId" , right_on="movieId")  #Στο dataframe df3 προσθέτω ακόμα
                                                        #ένα πεδίο για κάθε γραμμή που είναι η βαθμολογία του χρήστη
                                                        #Για τον ίδιο λόγο με πριν έχω βάλει το how = left για να διασφαλίσω ότι στο df5
                                                        #θα περιλαμβάνονται και οι ταινίες οι οποίες δεν έχουν βαθμολογηθεί από τον ζητούμενο χρήστη                              

df5 = df5.drop(['userId', 'timestamp'], axis=1)     #Διαγράφω τις στήλες που δεν χρειάζομαι
df5.rename(columns = {"rating" : "userRating"},inplace = True)

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

df6 = df5;  
df6["metric"] = df6[["BM25rating","meanRating","userRating"]].apply(rating_function,axis=1) #Εφαρμόζω την συνάρτηση rating_function για κάθε
                                                                                            #γραμμή του df6 (axis = 1) περνώντας σαν όρισμα ένα vector
                                                                                            #που περιλαμβάνει τα δεδομένα με τα οποία θα υπολογιστεί
                                                                                            #η νέα μετρική

df7_final = df6.sort_values(["metric"],ascending=False) #Ταξινομώ το df6 με βάση τo metric κατά φθίνουσα σειρά

print(df7_final[["title","genres","metric"]])       #Εκτύπωση αποτελέσματος


# for i in range(len(df7_final)):
#     print(df7_final.iloc[i].title,df7_final.iloc[i].genres,df7_final.iloc[i].metric)





 

    
   

   




