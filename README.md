# Multi-Agent Autonomous Delivery System with A* Pathfinding

**Academic Project - COMP4105: Designing Intelligent Agents**  
University of Nottingham, UK

## Overview

This project implements a multi-agent autonomous delivery system using A* pathfinding algorithm to investigate how the number of drone agents affects overall delivery efficiency across different environment types when constrained by battery limitations.

## Project Description

The simulation features autonomous drone agents operating in a 2D grid environment with three different scenarios:
- **Urban**: 10x10 grid with high obstacle density and delivery points
- **Suburban**: 12x12 grid with medium obstacle density and delivery points  
- **Rural**: 15x15 grid with low obstacle density and delivery points

### Key Features

- **Intelligent Agents**: Drone agents using A* pathfinding algorithm for navigation
- **Battery Management**: Agents must return to charging stations when battery is low
- **Package Delivery**: Agents collect packages from depots and deliver to randomized locations
- **Multiple Environment Types**: Three distinct grid configurations to test scalability
- **Comprehensive Experiments**: Automated testing across different agent counts (1, 3, 5, 8) with multiple trials

### Technologies Used

- **Python**: Core implementation language
- **Tkinter**: GUI for environment visualization
- **A\* Algorithm**: Pathfinding and navigation
- **Matplotlib & Pandas**: Data analysis and visualization
- **Threading**: Resource management and coordination
- **JSON**: Experimental data storage

## Experimental Results

The experiments demonstrate that increasing the number of agents significantly improves delivery efficiency across all environment types, with completion times reducing by approximately 84% when scaling from 1 to 8 agents.

## Current Limitations & Future Work

**Note**: Collision avoidance between agents is not currently implemented in this version. The A* algorithm works well for single-agent pathfinding but doesn't handle multi-agent collision detection. Future improvements will include:

- Implementation of Conflict-Based Search (CBS) or Improved Conflict-Based Search (ICBS) algorithms
- Enhanced multi-agent coordination strategies
- Dynamic pathfinding with real-time collision avoidance

## Running the Simulation

```bash
python main.py
```

The simulation will automatically run experiments across all environment types and agent configurations, generating results and visualizations.

## Results

Results are automatically saved as:
- `results.json`: Raw experimental data
- `complete_time_agent_graph.png`: Performance visualization
- `experiment_metrics.html`: Detailed performance metrics table

---

*This project was completed as coursework for COMP4105 - Designing Intelligent Agents at the University of Nottingham, investigating multi-agent systems and pathfinding algorithms in constrained environments.*
