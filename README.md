# LING 539 Kaggle Competition Code

This repository contains my code for the LING 539 Spring 2026 class-wide Kaggle competition.

## My submission

### Task

The task is a 3-way text classification problem:

- 0 = not a movie/TV review
- 1 = positive movie/TV review
- 2 = negative movie/TV review

### Approach

I represented the input text using TF-IDF features with unigram and bigram representations and compared three classifiers:

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVC

I evaluated the models using macro F1 on a held-out development split and selected the best model for the final Kaggle submission.

### Results

Local development macro F1 scores:

- Logistic Regression: 0.9142
- Multinomial Naive Bayes: 0.8851
- Linear SVC: 0.9192

The final submitted model was **Linear SVC**.

### How to run

Install dependencies:

```bash
pip install pandas scikit-learn
--

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/uhZ6joRH)
# Task

The task is described at [https://uazhlt-ms-program.github.io/ling-539-competition-2026/assignments/class-competition/](https://uazhlt-ms-program.github.io/ling-539-competition-2026/assignments/class-competition/)

The competition is hosted at [https://www.kaggle.com/competitions/ling-539-competition-2026](https://www.kaggle.com/competitions/ling-539-competition-2026)

**To join the competition, you must accept it at the following URL**: [https://www.kaggle.com/t/03c8dd2e91474ec1b64203601079805b](https://www.kaggle.com/t/03c8dd2e91474ec1b64203601079805b)

# Notes
- This project involves a **performance evaluation** as well as your **graded assessment**. It's important to keep these two things separate in your mind.
  - The rubric which will be used to assess your submission *for a grade* (ie, not to evaluate the performance of your model) is in the D2L assignment item
  - You are permitted to propose more than one classification model or approach. However, as described on the assessment rubric, **at least one of your submitted models must use one or more of the classification algorithms covered in this course.** (For more details related to assessment, be sure you understand the details of that rubric)
  - The performance of your model will be evaluated by Kaggle, and your model's performance will be ranked against other class submissions. The performance of your model is **one**, but not the only, factor by which your model will be assessed for a grade
- You are encouraged, but not obligated, to use Python
- You may delete or alter any files in this repository
- You are free to add dependencies, **however**, ensure that your code can be installed/used on another machine running Linux or MacOS (consider containerizing your project with Docker or an equivalent technology)
