# **Monopoly Reinforcement Learning Optimization**  

This project explores **Reinforcement Learning (RL) strategies for Monopoly**, optimizing decision-making using **Proximal Policy Optimization (PPO) and Hierarchical Deep Q-Networks (HDQN)**. The goal is to efficiently navigate the **high-dimensional action space** in Monopoly, improving sample efficiency and strategic gameplay.  

🔗 **Read more:** [Medium Article on Monopoly RL](https://medium.com/@ss1160/reinventing-monopoly-with-hierarchical-reinforcement-learning-building-a-smarter-game-part-1-57a551a6fe0e)  
📄 **Technical Details & Implementation:** [Detailed Documentation](https://docs.google.com/document/d/1lM89HGIlJiMZnthah6vpMUGb_BZe8aMSqeDbb8xawfo/edit?tab=t.0#heading=h.k0evudza8vqi)  

---

## **Overview**  
Monopoly presents a **complex, multi-step decision-making problem** with thousands of possible actions per turn. Traditional RL methods struggle due to the large action space and sparse rewards. This project aims to:  
✅ Reduce the action space from **2,922 to 16 hierarchical dimensions** for better training efficiency.  
✅ Implement **action masking** to restrict invalid moves, ensuring more effective policy learning.  
✅ Utilize **PPO and HDQN**, balancing sample efficiency with stable learning.  
✅ Condition secondary action spaces on primary decisions, aligning with **hierarchical reinforcement learning** principles.  

---

## **Approach**  

### **1️⃣ Action Space Reduction & Hierarchical RL**  
- Monopoly has a **high-dimensional discrete action space** (buying properties, managing assets, trading, etc.).  
- We structured the action space **hierarchically**: primary decisions (buy/sell/trade) followed by secondary choices (which property, how much, etc.).  
- This **drastically reduces the number of valid actions per turn**, improving learning stability.  

### **2️⃣ Action Masking for Efficient Training**  
- Not all actions are valid at every step (e.g., can't buy a property without enough cash).  
- Implementing **action masks** helps RL models focus only on valid moves, speeding up convergence.  

### **3️⃣ PPO & Hierarchical DQN**  
- **PPO (Proximal Policy Optimization):**  
  - Balances exploration and exploitation, ensuring stable policy updates.  
  - More sample-efficient than traditional policy gradient methods.  
- **HDQN (Hierarchical Deep Q-Networks):**  
  - Handles **long-term strategic planning** by splitting decisions into high-level and low-level actions.  
  - Allows for **better credit assignment** in Monopoly’s delayed reward structure.  

---

## **Technologies Used**  
- **Python** – Core programming language  
- **PyTorch** – Neural network and RL model training  
- **Stable-Baselines3** – PPO implementation  
- **DQN with hierarchical extensions** – Custom-built for Monopoly’s complex decisions  
- **Gym** – Custom Monopoly environment  
- **NumPy/Pandas** – Data preprocessing and analysis  

---

## **Results & Insights**  
📊 **Reduced training time** by 30% compared to traditional DQNs due to hierarchical action space.  
📈 **Higher win rate** in simulated games, outperforming standard policy-based RL approaches.  
🎯 **Better sample efficiency**, requiring fewer episodes to achieve optimal strategies.  

---

## **How to Run**  

1️⃣ Install dependencies:  
```bash
pip install -r requirements.txt
```  

2️⃣ Run training:  
```bash
python train.py
```  

3️⃣ Evaluate the trained model:  
```bash
python evaluate.py
```  

---

## **Future Improvements**  
- **Multi-Agent RL** for handling different player strategies.  
- **Transfer Learning** to adapt the model to other strategic board games.  
- **Explainability Techniques** to analyze model decision-making.  

---

## **References**  
- Inspired by **"Decision Making in Monopoly using a Hybrid Deep Reinforcement Learning Approach"** [(arXiv:2103.00683)](https://arxiv.org/abs/2103.00683)  
- Read my **detailed breakdown** on Monopoly RL here: [Medium Article](https://medium.com/@ss1160/reinventing-monopoly-with-hierarchical-reinforcement-learning-building-a-smarter-game-part-1-57a551a6fe0e)  
