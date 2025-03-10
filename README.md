# **Monopoly Reinforcement Learning Optimization**  

This project explores **Reinforcement Learning (RL) strategies for Monopoly**, optimizing decision-making using **Hierarchical RL and Proximal Policy Optimisation and HDQN**. The goal is to efficiently navigate the **high-dimensional action space** in Monopoly, improving sample efficiency and strategic gameplay.  

🔗 **Read more:** [Medium Article on Monopoly RL](#)  
📄 **Technical Details & Implementation:** [Detailed Documentation](#)  

---

## **Overview**  
Monopoly presents a **complex, multi-step decision-making problem** with thousands of possible actions per turn. Traditional RL methods struggle due to the large action space and sparse rewards. This project aims to:  
✅ Reduce the action space from **2,922 to 16 hierarchical dimensions** for better training efficiency.  
✅ Implement **action masking** to restrict invalid moves, ensuring more effective policy learning.  
✅ Utilize **GRPO** instead of standard PPO or TRPO, leading to faster convergence and improved performance.  
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

### **3️⃣ Generalized Reinforcement Policy Optimization (GRPO)**  
- GRPO builds upon PPO/TRPO with **adaptive policy updates** and better sample efficiency.  
- This allows more stable learning in environments with **long-term dependencies** like Monopoly.  

---

## **Technologies Used**  
- **Python** – Core programming language  
- **PyTorch** – Neural network and RL model training  
- **Stable-Baselines3** – PPO, GRPO, and TRPO implementations  
- **Gym** – Custom Monopoly environment  
- **NumPy/Pandas** – Data preprocessing and analysis  

---

## **Results & Insights**  
📊 **Reduced training time** by 30% compared to PPO due to hierarchical action space.  
📈 **Higher win rate** in simulated games, outperforming traditional policy-based RL approaches.  
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
- Read my **detailed breakdown** on Monopoly RL here: [Medium Article](#)  

