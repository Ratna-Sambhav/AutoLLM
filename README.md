# Project Proposal

## An AI Platform for Fine-Tuning, Deploying, and Inferencing on LLMs with RAG

**Project Lead:** Rahul Dave

**Authors:**
- Ratna Sambhav (sambhav.ratna@gmail.com)
- Prayash Panda (pkpanda234@gmail.com)
- Arjun Benoy (arjunbinoy.vishnu@gmail.com)
- Joyson Chacko George (joysoncgeorge2001@gmail.com)
- Praveen Kumar Vijayan
- Tarun Thakur (tarunsinghthakur.rst@gmail.com)

## Background and Motivation

With the large-scale industry adoption of large language models, there is a high demand to train custom LLMs on private data. Proprietary LLM APIs often lack data privacy and secure deployment, making it challenging for non-tech companies to adopt this technology. This project aims to bridge this gap by introducing a "no-code platform" that allows non-tech professionals to easily train and deploy their own LLMs, ensuring data privacy and cost-effectiveness.

## Scope and Objectives

The lack of a specialized no-code AI platform for non-tech personnel to train their LLMs is a challenge. Our AI platform, with an intuitive and easy user interface, aims to provide all the support needed for users to fine-tune their models effortlessly. The objectives include:

- Create automated pipelines for Fine-tuning, Model Deployment, and Inference.
- Develop a ReactJS frontend with a user login feature, interacting with a FastAPI backend.
- Utilize Infrastructure as Cloud service for resource-intensive tasks on GCP/AWS cloud.
- Showcase the system's functionality by fine-tuning and deploying 2b and 7b LLM models.

## Data Sources

**Source of Data:**
- Math-Instruct and Arithmo-data datasets from HuggingFace (Open source datasets).
- EDGAR Database (Accessible on www.sec.gov/edgar).

**Description of Datasets:**
- Math-Instruct and Arithmo-data: Question-Answer datasets with queries and responses in chain-of-thought format.
- EDGAR Database: A comprehensive online repository of corporate financial filings, accessible through APIs.

**Key Attributes:**
- Math-Instruct: MCQ type math-related questions and answers.
- Arithmo-data: Algebra-based questions and solutions.
- EDGAR Database: Text-based large set of documents.

**Relevance to the Project:**
- Math-Instruct and Arithmo-data for fine-tuning LLMs with mathematical skills.
- EDGAR Database to validate the correct working of the RAG pipeline.

## Data Handling after Building the Website and Pipelines

Throughout the three stages of model training, users can choose data from provided lists or upload their own. For Unsupervised and Supervised learning, users can upload documents or choose from predefined sets.

## Scope and Preliminary Design

Our AI platform will have a clean and responsive frontend, a highly scalable backend using container orchestration, and an integrated database. The system will be deployed on AWS.

