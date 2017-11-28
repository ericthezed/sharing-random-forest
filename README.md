# sharing-random-forest
Predicting Facebook Sharing

A Random Forest model was used to train and test several factors to determine the best predictors of whether or not a user would share an article on Facebook. Such factors included # of previous sessions, average # of pageviews per previous session, # of pageviews in current session, previous share count, previous pageview count, article shareability, article pageviews, whether or not the user had previously subscribed, and time since the user’s first visit.

It was found the the best predictors of sharing were number of shares in previous sessions, total number of previous sessions, article shareability, whether the user was on a mobile device, and time (in days) since the user's first visit. The model yielded a misclassification rate of 36.8%, which means sharing was correctly predicted 63.2% of the time. 
