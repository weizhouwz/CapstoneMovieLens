##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

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

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
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

##########################################################
# Data Processing and Exploration
##########################################################

# explore the dimension of the validation set
dim(validation)

# explore the dimension of the edx set, report unique counts of users and movies as well as the range of ratings
dim(edx)
n_distinct(edx$userId)
n_distinct(edx$movieId)
min(edx$rating)
max(edx$rating)

# explore the distribution of ratings
hist(edx$rating, main="Distrubution of Ratings", xlab="Rating", col="skyblue")

# see the first 6 rows of the data
head(edx)

# convert timestamp and create the difference in years variable in edx and validation sets
library(lubridate)
edx <- edx %>% mutate(timestamp=as_datetime(timestamp), yr=year(timestamp)) %>% extract(title, c("movietitle", "movieyear"), regex="^(.*) \\(([0-9]*)\\)$", remove=F) %>% mutate(yrdiff=ifelse(yr-as.numeric(movieyear)<0,0,yr-as.numeric(movieyear)))
validation <- validation %>% mutate(timestamp=as_datetime(timestamp), yr=year(timestamp)) %>% extract(title, c("movietitle", "movieyear"), regex="^(.*) \\(([0-9]*)\\)$", remove=F) %>% mutate(yrdiff=ifelse(yr-as.numeric(movieyear)<0,0,yr-as.numeric(movieyear)))

##########################################################
# Create training and test datasets
##########################################################

# set seed to 2
set.seed(2,sample.kind="Rounding")

# create training and test sets with test set being 11% of the edx set
ind <- createDataPartition(y=edx$rating, times=1, p=0.11, list=FALSE)
traindata <- edx[-ind,]
tmp <- edx[ind,]

# make sure userId and movieId in test set are also in training set
testdata <- tmp %>% semi_join(traindata, by="movieId") %>% semi_join(traindata, by="userId")

# check the number of records in the test set
nrow(testdata)

# add rows removed back into the training set
extra <- anti_join(tmp, testdata)
traindata <- rbind(traindata, extra)

##########################################################
# Model Development
##########################################################

#############################
##### 1. Original Model #####

# create the original model and tune it to find the best value for lambda
lambdas <- seq(0, 10, 0.25)
rmses_orig <- sapply(lambdas, function(l){
	mu <- mean(traindata$rating)
	b_i <- traindata %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n()+l))
	b_u <- traindata %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarize(b_u = sum(rating - b_i - mu)/(n()+l))
	predicted_ratings <- testdata %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>% mutate(pred = mu + b_i + b_u) %>% pull(pred)
	sqrt(mean((testdata$rating - predicted_ratings)^2))
})
min(rmses_orig)
lambda_orig <- lambdas[which.min(rmses_orig)]
lambda_orig

# create the model with the best lambda on the edx set and calculate the RMSE using the validation set
lambda <- lambda_orig
mu_edx <- mean(edx$rating)
b_i <- edx %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu_edx)/(n()+lambda))
b_u <- edx %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarize(b_u = sum(rating - b_i - mu_edx)/(n()+lambda))
r_hat <- validation %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>% mutate(pred = mu_edx + b_i + b_u) %>% pull(pred)
rmse_orig <- sqrt(mean((validation$rating - r_hat)^2))

#############################
##### 2. New Model I #####

# check the genres effect using lambda from the original model
lambda <- lambda_orig
mu <- mean(traindata$rating)
b_i <- traindata %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n()+lambda))
b_u <- traindata %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
b_g <- traindata %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% group_by(genres) %>% summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+lambda))

# explore the distribution of the genres effect
hist(b_g$b_g, main="Distribution of Genres Effect", xlab="Genres Effect", col="skyblue")

# create New Model I and tune it to find the best value for lambda
lambdas <- seq(0, 10, 0.25)
rmses_m1 <- sapply(lambdas, function(l){
	mu <- mean(traindata$rating)
	b_i <- traindata %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n()+l))
	b_u <- traindata %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarize(b_u = sum(rating - b_i - mu)/(n()+l))
	b_g <- traindata %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% group_by(genres) %>% summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
	predicted_ratings <- testdata %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>% left_join(b_g, by="genres") %>% mutate(pred = mu + b_i + b_u + b_g) %>% pull(pred)
	sqrt(mean((testdata$rating - predicted_ratings)^2))
})
min(rmses_m1)
lambda_m1 <- lambdas[which.min(rmses_m1)]
lambda_m1

# create New Model I with the best lambda on the edx set and calculate the RMSE using the validation set
lambda <- lambda_m1
mu_edx <- mean(edx$rating)
b_i <- edx %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu_edx)/(n()+lambda))
b_u <- edx %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarize(b_u = sum(rating - b_i - mu_edx)/(n()+lambda))
b_g <- edx %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% group_by(genres) %>% summarize(b_g = sum(rating - b_i - b_u - mu_edx)/(n()+lambda))
r_hat <- validation %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>% left_join(b_g, by="genres") %>% mutate(pred = mu_edx + b_i + b_u + b_g) %>% pull(pred)
rmse_m1 <- sqrt(mean((validation$rating - r_hat)^2))

#############################
##### 3. New Model II #####

# explore relationship between difference in years and average rating
edx %>% group_by(yrdiff) %>% summarize(avg=mean(rating)) %>% ggplot(aes(yrdiff, avg)) + geom_point(color="blue") + ggtitle("Plot Average Rating by Difference in Years") + xlab("Difference in Years") + ylab("Average Rating")

# check the time effect using lambda from the original model
lambda <- lambda_orig
mu <- mean(traindata$rating)
b_i <- traindata %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n()+lambda))
b_u <- traindata %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
b_g <- traindata %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% group_by(genres) %>% summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+lambda))
b_y <- traindata %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_g, by="genres") %>% group_by(yrdiff) %>% summarize(b_y = sum(rating - b_i - b_u - b_g - mu)/(n()+lambda))

# explore the distribution of the time effect
hist(b_y$b_y, main="Distribution of Time Effect", xlab="Time Effect", col="skyblue")

# create New Model II and tune it to find the best value for lambda
lambdas <- seq(0, 10, 0.25)
rmses_m2 <- sapply(lambdas, function(l){
	mu <- mean(traindata$rating)
	b_i <- traindata %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n()+l))
	b_u <- traindata %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarize(b_u = sum(rating - b_i - mu)/(n()+l))
	b_g <- traindata %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% group_by(genres) %>% summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
	b_y <- traindata %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_g, by="genres") %>% group_by(yrdiff) %>% summarize(b_y = sum(rating - b_i - b_u - b_g - mu)/(n()+l))
	predicted_ratings <- testdata %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>% left_join(b_g, by="genres") %>% left_join(b_y, by="yrdiff") %>% mutate(pred = mu + b_i + b_u + b_g + b_y) %>% pull(pred)
	sqrt(mean((testdata$rating - predicted_ratings)^2))
})
min(rmses_m2)
lambda_m2 <- lambdas[which.min(rmses_m2)]
lambda_m2

# create New Model II with the best lambda on the edx set and calculate the RMSE using the validation set
lambda <- lambda_m2
mu_edx <- mean(edx$rating)
b_i <- edx %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu_edx)/(n()+lambda))
b_u <- edx %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarize(b_u = sum(rating - b_i - mu_edx)/(n()+lambda))
b_g <- edx %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% group_by(genres) %>% summarize(b_g = sum(rating - b_i - b_u - mu_edx)/(n()+lambda))
b_y <- edx %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_g, by="genres") %>% group_by(yrdiff) %>% summarize(b_y = sum(rating - b_i - b_u - b_g - mu_edx)/(n()+lambda))
r_hat <- validation %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>% left_join(b_g, by="genres") %>% left_join(b_y, by="yrdiff") %>% mutate(pred = mu_edx + b_i + b_u + b_g + b_y) %>% pull(pred)
rmse_m2 <- sqrt(mean((validation$rating - r_hat)^2))

##########################################################
# Print RMSEs and Select Model
##########################################################

# print the RMSE of the original model
cat("RMSE of the original model: ", rmse_orig, "\n")

# print the RMSE of New Model I
cat("RMSE of New Model I: ", rmse_m1, "\n")

# print the RMSE of New Model II
cat("RMSE of New Model II: ", rmse_m2, "\n")
