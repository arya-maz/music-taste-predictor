# Music Taste Predictor

A personal machine learning project that analyzes my 2025 album listening dataset and predicts how much I may enjoy new albums based on attributes such as artist, release year, track count, runtime, genre tags, and listening score.

## Background

As a long-time avid music fanatic, I've always been eagerly chasing new experiences in the form of exploring new artists, genres, styles, etc. Throughout 2025, I challenged myself to check out 365 albums that I had never heard before -- some of which were newer albums releasing on a rolling basis throughout the year, while the majority were either releases that I had missed from recent years, or relics of older eras that I wanted to closely familiarize myself with.

## Project Goals

- Build a personal music taste profile from a CSV of album ratings
- Analyze genre, runtime, track count, and release-year patterns
- Train a machine learning model to predict album scores
- Recommend unlistened albums based on learned preferences
- Eventually turn the project into a full web application

## Current Status

This project is in the early setup and data exploration phase.

## Dataset

The dataset contains 365 albums listened to for the first time throughout 2025, including:

- Artist
- Album title
- Release year
- Number of tracks
- Runtime in minutes
- Average track length
- Three genre tags
- Personal score out of 100

## Tech Stack

- Python
- pandas
- scikit-learn
- CatBoost
- FastAPI
- PostgreSQL
- React frontend

## Roadmap

- [x] Clean album dataset
- [x] Set up Python project structure
- [ ] Perform exploratory data analysis
- [ ] Engineer model features
- [ ] Train baseline prediction model
- [ ] Build score prediction script
- [ ] Build recommendation system
- [ ] Create interactive prototype
- [ ] Expand into full web application
