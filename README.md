# Snake AI Game

Welcome to the Snake AI Game! This project implements an AI agent that learns to play the classic Snake game using reinforcement learning.

## Table of Contents

- [Snake AI Game](#snake-ai-game)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Game Overview](#game-overview)
  - [Installation](#installation)
  - [Usage](#usage)

## Introduction

This project showcases the implementation of an AI agent trained to play the Snake game using reinforcement learning. The agent learns to make decisions to control the snake's movements in order to maximize its score by consuming food items while avoiding collisions with walls and its own body.

## Game Overview

The Snake AI game consists of two main components:

1. **Snake Game Environment (game.py):**
   This module provides the game logic, rendering, and interaction with the game state. The snake's movements are controlled by the AI agent, and the agent's decisions influence the game's progression.

2. **AI Agent (agent.py and model.py):**
   The AI agent learns to play the game using Q-learning, a reinforcement learning algorithm. The agent's decisions are based on the game's state, and it learns to take actions that maximize cumulative rewards over time.

## Installation

1. **Clone the repository:**
```sh
git clone https://github.com/yourusername/snake-ai.git
```
2. **Install the required dependencies:**
```sh
pip install -r requirements.txt
```

## Usage

1. **Run the Snake AI training script:**
```sh
python agent.py
```
This script will train the AI agent to play the Snake game using reinforcement learning.

2. **Observe the agent's progress and performance in the game. The agent's learning progress will be displayed in the console.**

3. **Once training is complete, the agent's model will be saved for future use.**