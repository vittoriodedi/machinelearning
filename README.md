# machinelearning
This repository contains my 'Machine Learning' course final project and some other stuff

## Overview

The project explores the application of **Reinforcement Learning** and **Deep Q-Learning** in a custom game environment called **"Cat-Mouse-Cheese"**. The goal is for the mouse agent to reach the cheese while avoiding the cat, learning optimal movement strategies across different grid configurations. The development involved both classic **Q-learning** and **Deep Q-Networks (DQN)** for comparison on various map complexities.

## Contents

* 📁 **`DQN/`** – Contains the implementation of the **10x10 grid** with obstacles using **Deep Q-Learning**. This version uses a neural network to approximate Q-values, allowing the agent to learn in a more complex and continuous state space.

* 📁 **`RL Q Tables/`** – Core project using classic **Q-learning** with Q-tables:

  * 📁 `principale/` – The **main implementation** built collaboratively by the group. It includes:

    * 5x5 empty grid
    * 5x5 grid with walls
    * 10x10 grid with obstacles
      These environments implement a high-dimensional Q-table, allowing fine-grained control and evaluation of the agent’s learning across scenarios.

  * 📁 `alternativo/` – An **alternative version** developed by one group member. It features:

    * Custom 10x10 grid with **"deadly trap" mechanics**
    * Variants with and without the presence of the cat
      This version explores experimental changes to the environment and reward logic.

* 📁 **`esposizione/`** – Files sourced from a recommended GitHub repository (as suggested by the professor). These were used for classroom presentation and **not written by me**.

* 📄 **`Relazione Machine Learning.pdf`** – The final report documenting the **complete development cycle**, including:

  * Theoretical background on Reinforcement Learning and Q-learning
  * Description of all implemented environments
  * Reward design and training strategy
  * Graphical visualization and performance analysis
  * Comparison between the different learning approaches and environments

## Purpose

This repository showcases our collaborative work in building intelligent agents capable of learning optimal strategies through experience. The project focuses on:

* Applying and comparing Q-learning vs. Deep Q-Learning
* Custom environment design with increasing difficulty
* Implementing reward-based learning logic
* Graphical simulation and real-time rendering with **Pygame**
* Tracking and analyzing performance over time