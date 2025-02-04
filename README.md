# AgentCF++
 
AgentCF++ is a Shared Memory Enhanced Collaborative Learning framework with LLMs-Powered Agents designed for Cross-Domain Recommender Systems.
 
# Table of Contents
 
- [AgentCF++](#agentcf)
- [Table of Contents](#table-of-contents)
- [Description](#description)
- [Usage](#usage)
- [Data Source](#data-source)
- [Complete Results](#complete-results)
 
# Description
 
AgentCF++ leverages advanced collaborative learning techniques to enhance recommendation systems across different domains. It utilizes large language models (LLMs) to empower agents in providing personalized recommendations.
 
# Usage
 
1. **Data Processing**:
   ```bash
   python dataprocess/crossDomainDataPrepare.py
2. **Training**:
   ```bash
   python AgentCF++.py
3. **Testing**:
   ```bash
   python AgentCF++Test.py
Run the code in the user_group_mem directory to group users based on their interests.

# Data Source
The data used for this project can be found at: [(https://amazon-reviews-2023.github.io/main.html)]

# Complete Results
**MRR:**

| Method               | Cross-1 | Cross-2 | Cross-3 | Cross-4 | Cross-5 |
| -------------------- | ------- | ------- | ------- | ------- | ------- |
| BPR-MF               | 0.2949  | 0.2959  | 0.3114  | 0.3012  | 0.3127  |
| SASRec               | 0.3463  | 0.3154  | 0.3828  | 0.3118  | 0.3687  |
| Pop                  | 0.2589  | 0.2817  | 0.3094  | 0.2954  | 0.3089  |
| LLMSeqSim            | 0.2646  | 0.2549  | 0.3101  | 0.2959  | 0.3124  |
| LLMRank              | 0.3268  | 0.2730  | 0.3106  | 0.2970  | 0.3308  |
| AgentCF              | 0.3284  | 0.2681  | 0.3114  | 0.3032  | 0.3480  |
| AgentCF++            | 0.3537  | 0.3176  | 0.3989  | 0.3321  | 0.3837  |
| AgentCF  + dual      | 0.3495  | 0.2962  | 0.3581  | 0.3139  | 0.3581  |
| AgentCF  + shared    | 0.3488  | 0.2777  | 0.3190  | 0.3147  | 0.3689  |
| AgentCF++  w/o group | 0.3415  | 0.2724  | 0.3181  | 0.3126  | 0.3549  |

**NDCG:**

| Method                | Cross-1 | Cross-2 | Cross-3 | Cross-4 | Cross-5 |
| --------------------- | ------- | ------- | ------- | ------- | ------- |
| BPR-MF                | 0.4566  | 0.4577  | 0.4697  | 0.4612  | 0.4695  |
| SASRec                | 0.4949  | 0.4707  | 0.5244  | 0.4684  | 0.5143  |
| Pop                   | 0.4284  | 0.4463  | 0.4691  | 0.4587  | 0.4673  |
| LLMSeqSim             | 0.4316  | 0.4246  | 0.4693  | 0.4587  | 0.4711  |
| LLMRank               | 0.4818  | 0.4393  | 0.4694  | 0.4589  | 0.4853  |
| AgentCF               | 0.4829  | 0.4343  | 0.4697  | 0.4636  | 0.4984  |
| AgentCF++             | 0.5020  | 0.4728  | 0.5360  | 0.4880  | 0.5268  |
| AgentCF  + dual       | 0.4997  | 0.4552  | 0.5053  | 0.4732  | 0.5072  |
| AgentCF  + shared     | 0.4974  | 0.4432  | 0.4741  | 0.4723  | 0.5146  |
| AgentCF  ++ w/o group | 0.4936  | 0.4380  | 0.4732  | 0.4703  | 0.5040  |