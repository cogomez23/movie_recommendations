#Note: I'm using the most updated R version as of March 2021.
library(dplyr)
library(dslabs)
options(digits = 5)

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

validation <- read.csv(file = "validation.csv")

#Our goal is to predict the rating for a particular movie and from a particular 
#  user and we want to create an effective algorithm by minimizing the 
#  Residual Mean Squared Error(RMSE). Edx is our training set
#  while validation is our test_set
#We will create an RMSE function with this:
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Step 1. . Since we cannot use the test set in choosing
#  the algorithm, we will further divide the training set
#  edx into a train_set and test_set. We will allocate
#  20 percent of the data to the test_set
set.seed(1, sample.kind = "Rounding") #This would be set.seed(1) for older versions of R
test_index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)
test_set <- edx %>% slice(test_index)
train_set <- edx %>% slice(-test_index)
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")       #The last 2 lines are to make sure there's the same userId and movieId

#We will now focus on training on the train_set and find out the 
#  algorithm that minimizes the Residual Mean Squared Error(RMSE)

#Step 2. We want to create a baseline rating. That can be found
#  by calculating the mean of the ratings across all movies

mu <- mean(train_set$rating)
mu

#We know the mean mu is 3.5125. We can use this as a starting point for
#  our prediction. We can do a quick check with our RMSE:
RMSE(test_set$rating, mu)

#An RMSE of 1.0599 is bad because we are missing by 1 point on average. There
#  is a big room for improvement

#Now we will factor in the movie effect. Some movies will naturally
#  be rated higher due to various reasons such as quality, and some will
#  be rated lower. We can easily do this by mutating the training set. 
#A better way is to run a linear regression but with 9 million
#  observations, this will take a long time and I do not have the 
#  hardware capabilities

#Step 3:
#movie_effect can be calculated by finding the mean of the difference between 
#  the rating and mu

movie_ave <- train_set %>% group_by(movieId) %>% 
  summarize(movie_effect = mean(rating - mu)) 

#Let's do a quick check on our RMSE. We will add mu and the movie effect
predicted_ratings_1 <- mu + test_set %>% left_join(movie_ave, by='movieId') %>%.$movie_effect
RMSE(test_set$rating, predicted_ratings_1)

#We have improve our model and the RMSE is now 0.94374

#Step 4. Now we want to find the user effect. Some people will rate movies  
#  higher than most people and some will rate movies lower. We will quantify
#  this bias. Again, it is better to run a linear regression but we
#  can do the same as step 3.

user_ave <- train_set %>% 
  left_join(movie_ave, by="movieId") %>% group_by(userId) %>%
  summarize(user_effect = mean(rating - mu - movie_effect))

predicted_ratings_2 <- test_set %>% 
  left_join(user_ave, by="userId") %>% 
  left_join(movie_ave, by="movieId") %>%
  mutate(predictions = mu + movie_effect + user_effect) %>% .$predictions
  
RMSE(test_set$rating,predicted_ratings_2)

#Now we have an RMSE of 0.86593 which is even better.


#Step 5. Implement Regularization
#A quick look at the data with the following code shows that most of the best 
#  and worst movies were rated not more than 5 times. We can check that through
#  these codes:

movie_titles <- edx  %>% 
  select(movieId, title) %>%
  distinct()

train_set %>% count(movieId) %>% 
  left_join(movie_ave) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(movie_effect)) %>% 
  select(title, movie_effect, n) %>% 
  slice(1:10)

train_set %>% count(movieId) %>% 
  left_join(movie_ave) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange((movie_effect)) %>% 
  select(title, movie_effect, n) %>% 
  slice(1:10)

#In order to improve our model, we need to penalize movies that have lower
#  number of ratings. We can do this through regularization. Basically we want
#  to divide the movie effect and user effect with the number of ratings plus 
#  an added value lambda instead of simply taking the mean.
#  This is easier to understand once we apply this. This is the code to find
#  the optimal lambda and the RMSE associated with it
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(x){
  movie_ave_reg <- train_set %>% group_by(movieId) %>% 
    summarize(movie_effect = sum(rating - mu)/(n()+x)) 
  user_ave_reg <- train_set %>% 
    left_join(movie_ave_reg, by="movieId") %>% group_by(userId) %>%
    summarize(user_effect = sum(rating - mu - movie_effect)/(n()+x))
  predicted_ratings_3 <- test_set %>% 
    left_join(movie_ave_reg, by = "movieId") %>%
    left_join(user_ave_reg, by = "userId") %>%
    mutate(predictions = mu + movie_effect + user_effect) %>%
    .$predictions
  RMSE(test_set$rating, predicted_ratings_3)
})
lambdas[which.min(rmses)]
min(rmses)

#We see that the value of lambda that minimizes the code is 4.75
#  This gives us an RMSE of 0.86524 which is only slightly better than 
#  0.86593, but is still is a significant improvement still. 
#  This is our current code:
movie_ave_reg <- train_set %>% group_by(movieId) %>% 
  summarize(movie_effect = sum(rating - mu)/(n()+4.75)) 
user_ave_reg <- train_set %>% 
  left_join(movie_ave_reg, by="movieId") %>% group_by(userId) %>%
  summarize(user_effect = sum(rating - mu - movie_effect)/(n()+4.75))
predicted_ratings_3 <- test_set %>% 
  left_join(movie_ave_reg, by = "movieId") %>%
  left_join(user_ave_reg, by = "userId") %>%
  mutate(predictions = mu + movie_effect + user_effect) %>%
  .$predictions
RMSE(test_set$rating, predicted_ratings_3)

#Calculate for a genre effect. We do not need to regularize since the genres
#  tally does not work the same way as the userId's and movieId's.
#  Checking for a genre effect. We do this through this code:
edx %>% group_by(genres) %>%
  summarize(n = n(), ave = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 100000) %>% 
  mutate(genres = reorder(genres, ave)) %>%
  ggplot(aes(x = genres, y = ave, ymin = ave - 2*se, ymax = ave + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
#As we can see, among the most rated genres, Comedy has the lowest average 
#  rating while Drama/War, gets rated the highest.  
#  We can calculate for the genre effect through this code:
genre_ave <- train_set %>% 
  left_join(user_ave_reg, by="userId") %>% 
  left_join(movie_ave_reg, by = "movieId") %>%
  group_by(genres) %>%
  summarize(genre_effect = mean(rating - mu - movie_effect - user_effect))

predicted_ratings_4 <- test_set %>% 
  left_join(user_ave_reg, by="userId") %>% 
  left_join(movie_ave_reg, by="movieId") %>%
  left_join(genre_ave, by= "genres") %>%
  mutate(predictions = mu + movie_effect + user_effect + genre_effect) %>% 
  .$predictions

RMSE(test_set$rating,predicted_ratings_4)

#With this, the RMSE drops to 0.86494

#A much better way is to group by c(userId, genres) giving us a much more
#  personalized result but my computer cannot process it. It is the same case
#  with more complex algorithms such as matrix factorization.

#This algorithm looks good enough. Now we will apply this on the validation set.
#   But first we want to update our algorithm to run on the entire edx set 
#   instead of just the train_set. More data is better for an algorithm.

movie_ave_reg_edx <- edx %>% group_by(movieId) %>% 
  summarize(movie_effect = sum(rating - mu)/(n()+4.75)) 
user_ave_reg_edx <- edx %>% 
  left_join(movie_ave_reg_edx, by="movieId") %>% group_by(userId) %>%
  summarize(user_effect = sum(rating - mu - movie_effect)/(n()+4.75))
genre_ave_edx <- edx %>% 
  left_join(user_ave_reg_edx, by="userId") %>% 
  left_join(movie_ave_reg_edx, by = "movieId") %>%
  group_by(genres) %>%
  summarize(genre_effect = mean(rating - mu - movie_effect - user_effect))

validation_predictions <- validation %>% 
  left_join(user_ave_reg_edx, by="userId") %>% 
  left_join(movie_ave_reg_edx, by="movieId") %>%
  left_join(genre_ave_edx, by= "genres") %>%
  mutate(predictions = mu + movie_effect + user_effect + genre_effect) %>% 
  .$predictions

validation_predictions[is.na(validation_predictions)] <- 0 # setting NAs to zero. If we don't remove this we also receive NA as our RMSE. There are only a handful of NAs so any rating value will do actually 

RMSE(validation$rating,validation_predictions)

#Our final RMSE is 0.86445