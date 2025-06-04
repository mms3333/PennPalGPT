Notebooks:

** feature_generation.ipynb - uses a gpt-3.5 model to generate features and use few-shot prompting with 3 examples of articles with extracted topics discussed in the research papers. The question 'What are key features of {professor's name} work?' is passed for every professor to get a large set of possible features that captures every professor. A couple different types of prompts were attempted but few shot prompt was found to be the most successful. Upon receiving these results, the dataframe 'professor_research_results.csv' was created with 'professor' and 'features' columns. 93 features were created from this list, coming up with a combination of broad and niche features. 

To run this notebook, please upload the pyproject.toml and TrainingDataSmall.zip file and run using gpus.

** grouping_professors.ipynb - run this notebook with the dataset 'data_professors.csv' to analyze how similar professors are in their interests. This notebook uses K-Means to find 6 groups that were later used in our score.py script that allows more context into how close the answers are to being correct.

To run this notebook, please upload 'data_professors.csv'

** feature_transformation_for_test.ipynb - In this notebook, a gpt-4 model and a prompt instruct the model to return 0 and 1 for all of the features. A matrix is then created for all of the question's features. A Random Forest classifier, SVC, and KNN model are also in this section. Both models are trained on the professor data and predict a professor from our question matrices. The files used to run this are 'dev.csv', our dev data, and data_professors.csv. Upon running this notebook, the output 'question_matrices.csv' consists the question matrices, 'results_random_forest.csv' consists the random forest results, 'results_svc' consists the SVC results, and 'results_knn.csv' consists the KNN results. Note that the SVC and KNN results consist a list of professors, which is what was initially proposed for the model to return.

Python files:

** score.py - evaluates the quality of the answer. 
