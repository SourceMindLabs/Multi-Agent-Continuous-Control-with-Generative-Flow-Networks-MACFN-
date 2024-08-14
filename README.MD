# Multi-Agent Continuous Control with Generative Flow Networks (MACFN)

## Overview

This project implements a framework for **Multi-Agent Continuous Control with Generative Flow Networks (MACFN)**. The primary goal of this project is to extend the capabilities of Generative Flow Networks (GFlowNets) to handle multi-agent systems in continuous control environments. The agents learn decentralized policies through centralized training, allowing them to perform cooperative exploration in tasks with continuous state and action spaces.

### Key Features:
- **Flow Matching Loss**: Implements a continuous flow matching loss function that ensures inflows and outflows are consistent with the theoretical framework.
- **Flow Decomposition Network**: Ensures the consistency between global and individual flows by decomposing the joint flow into agent-specific contributions.
- **Inverse Transition Network**: Estimates previous states given current states and actions, essential for calculating inflows.
- **Sampling-Based Approach**: Approximates integrals over inflows and outflows using sampled actions.
- **Centralized Training, Decentralized Execution (CTDE)**: Trains agents using global information and allows them to act based on local policies during execution.

## Paper Reference

This implementation is based on the following paper:

**Multi-Agent Continuous Control with Generative Flow Networks (MACFN)**

- Authors: Shuang Luo, Yinchuan Li, Shunyu Liu, Xu Zhang, Yunfeng Shao, Chao Wu
- Published by: Neural Networks (Accepted for publication)
- [ArXiv Link](https://arxiv.org/abs/2408.06920)
- [DOI Link](https://doi.org/10.1016/j.neunet.2024.106243)

The paper proposes a novel method for multi-agent systems in continuous environments using GFlowNets. It extends the flow matching concept to multi-agent scenarios, allowing for more effective exploration and learning in cooperative tasks.

## Project Structure

- `environment.py`: Defines the environment class where agents operate.
- `flow_network.py`: Contains the implementation of the flow network used by each agent.
- `inverse_transition_network.py`: Implements the inverse transition network for estimating parent states.
- `flow_decomposition_network.py`: Implements the flow decomposition network for separating global and individual flows.
- `macfn.py`: Contains the main MACFN class that integrates all components and handles the training process.
- `train.py`: The main script to run the training process.
