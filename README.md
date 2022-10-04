# DataMiningTechniques 2022
Our task is to predict what hotel a user is most likely to book. This could greatly help companies such as Expedia (from which the dataset actually originates) to organize the search results for a user in the most suitable way. This assignment was an internal competition, where my team ranked 2nd among 173 groups. 

## Detailed Task Description
To make things easier, we'll use a DM process model to describe your task in a bit more detail:

### 1. Business Understanding
Your task is to predict what hotels properties listed as a result of a hotel search a user is most likely to click on. Of course, more people have worked on such predictions. Can you find some other people that have tried to make such predictions (e.g. from the Kaggle competition)? And what have they used as most prominent predictors? Have other people that participate in the competition mentioned anything about their approaches? Please spend a couple of paragraphs on this topic (i.e. related work) in your report.

### 2. Data Understanding
Essentially, this is a sub-task that requires you do Exploratory Data Analysis (EDA). Explore the dataset, count, summarize, plot things, and report findings that are useful for your task. Remember that EDA is not necessary done once and then you move on. It might very well be possible that you do some EDA, build some models, then some idea comes up, do some more EDA, modify your model according to what it shows, and so on.

### 3. Data Preparation
You'll certainly need to work on the dataset, to create, modify or add new features. For instance, you might want to compare the different properties that resulted from the search instead of learning from them one by one. There are certain attributes with a large amount of missing values, do they still provide useful information? And how will you handle a missing value if this shows to be the case? Finally, in order to test your approach (since you do not know the answers for the test set) you will need to split up your data to test your approach yourself before you generate your answers on our test set. Of course, you are also allowed to use external data sources if you find ones that are useful.

### 4. Modeling and Evaluation
Naturally, once you prepare the dataset, you should be able to build models. Use whatever technique you want, though the choice might be influenced by how we would like to measure your predictions at the end (see Section 5.1). To test how your model is compared to other, you can upload your answers for the test set on this website . Note that the score shown there is only for part of the test set, the score on the rest of the test set will only be disclosed on the final lecture and will form part of your final grade.
