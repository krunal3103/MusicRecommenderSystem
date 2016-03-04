#Music Recommender System using Apache Spark and Python

# For this project, I have created a recommender system that will recommend new musical artists to a user based on their listening history. Suggesting different songs or musical artists to a user is important to many music streaming services, such as Pandora and Spotify. In addition, this type of recommender system could also be used as a means of suggesting TV shows or movies to a user (e.g., Netflix).
# To create this system I will be using Spark and the collaborative filtering technique.

# ## Datasets
# I have used some publicly available song data from audioscrobbler, which can be found [here](http://www-etud.iro.umontreal.ca/~bergstrj/audioscrobbler_data.html). However, I modified the original data files so that the code will run in a reasonable time on a single machine. The reduced data files have been suffixed with `_small.txt` and contains only the information relevant to the top 50 most prolific users (highest artist play counts).
# The `artist_data.txt` file then provides a map from the canonical artist ID to the name of the artist.

from pyspark.mllib.recommendation import *
import random
from operator import *
# ## Loading data

artistData= sc.textFile("artist_data_small.txt").map(lambda line:line.split("\t")).map(lambda x: (int(x[0]),x[1]))
artistAlias = sc.textFile("artist_alias_small.txt").map(lambda line:line.split("\t")).map(lambda x: (int(x[0]),int(x[1])))
userArtistData = sc.textFile("user_artist_data_small.txt").map(lambda line:line.split(" ")).map(lambda x: (int(x[0]),int(x[1]),int(x[2])))
# ## Data Exploration

p= userArtistData.map(lambda b: (b[0],(b[1],b[2]))).groupByKey().map(lambda b: (b[0],dict(b[1])))
p=p.map(lambda b: (b[0],sum(b[1].values()),sum(b[1].values())/len(b[1])))
p=p.sortBy(keyfunc=lambda b: b[1], ascending=False).take(3)
for i in p:
    print("User %d has a total play count of %d and a mean play count of %d") %(i[0],i[1],i[2])


# ####  Splitting Data for Testing
# Use the [randomSplit](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.randomSplit) function to divide the data (`userArtistData`) into:
trainData, validationData, testData= userArtistData.randomSplit((40,40,20), seed=13)
# ## The Recommender Model
#
# For this project, I will train the model with implicit feedback.
# ### Model Evaluation
#
# Although there may be several ways to evaluate a model, I will use a simple method here. Suppose we have a model and some dataset of *true* artist plays for a set of users. This model can be used to predict the top X artist recommendations for a user and these recommendations can be compared the artists that the user actually listened to (here, X will be the number of artists in the dataset of *true* artist plays). Then, the fraction of overlap between the top X predictions of the model and the X artists that the user actually listened to can be calculated. This process can be repeated for all users and an average value returned.
#
# For example, suppose a model predicted [1,2,4,8] as the top X=4 artists for a user. Suppose, that user actually listened to the artists [1,3,7,8]. Then, for this user, the model would have a score of 2/4=0.5. To get the overall score, this would be performed for all users, with the average returned.
#
def modelEval(bestModel,dataset=validationData):

    #find all artists and remove duplicates
    UserArtists= sc.parallelize(set(userArtistData.map(lambda b:b[1]).collect()))
    #find all users
    users= dataset.map(lambda b: (b[0],b[1])).groupByKey().map(lambda b:b[0])
    #create user artist pairs
    totalUserArtists= users.cartesian(UserArtists)

    totalUserArtistsTraining= trainData.map(lambda b:(b[0],b[1]))
    #generate user artist pairs to predict on
    predictOnUserArtists = totalUserArtists.subtract(totalUserArtistsTraining)

    predictedResult= bestModel.predictAll(predictOnUserArtists)

    sortedPredictedResult= predictedResult.sortBy(lambda k:(k.user, -k.rating))    .map(lambda b:(b[0],b[1]))
    sortedPredictedResult=sortedPredictedResult.groupByKey()    .map(lambda b:(b[0],list(b[1])))


    testDataArtistList=dataset.map(lambda b:(b[0],b[1])).groupByKey()    .map(lambda b:(b[0],list(b[1])))
    userXCount= dataset.map(lambda b:(b[0],b[1])).groupByKey()    .map(lambda b:(b[0],len(b[1])))


    p= userXCount.join(sortedPredictedResult)
    topGlobalArtistsX= p.map(lambda b:(b[0],b[1][1][0:b[1][0]]))
    #topGlobalArtistsX.collect()
    q=topGlobalArtistsX.join(testDataArtistList)
    q=q.map(lambda b: (b[0],set.intersection(set(b[1][0]),set(b[1][1]))))

    o= q.map(lambda b:(b[0], len(b[1])))

    o.collect()
    k= userXCount.join(o).map(lambda b:(b[0],b[1][1]/float(b[1][0])))
    k.collect()
    ans=0
    for i in k.collect():
        ans+=i[1]
    ans= ans/float(k.count())
    return ans


# Model Construction
ranks = [2, 10, 20]
for r in ranks:
   model = ALS.trainImplicit(trainData, rank=int(r), seed=345)
   score = modelEval(model)
   print("The model score for rank " + str(r) + " is " + str(score))

bestModel = ALS.trainImplicit(trainData, rank=10, seed=345)
Result=modelEval(bestModel, testData)
print Result

#One Example
artists= bestModel.recommendProducts(1059637,5)
artists=sc.parallelize(artists).map(lambda b:b.product)
ids=artists.collect()
c=0
for i in ids:
    for p in artistData.collect():
        if(p[0]==i):
            print ("Artist %d: "%c+p[1])
            c+=1
