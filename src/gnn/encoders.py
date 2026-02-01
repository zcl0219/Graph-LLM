import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
# 假设我们在项目根目录运行，使用 src. 前缀
from src.agents.base_agent import BaseAgent

class NodeEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # 只加载模型，不绑定具体数据
        self.model = SentenceTransformer(model_name)
    
    def encode_agents(self, agents: List[BaseAgent]) -> torch.Tensor:
        """
        Encode a list of agents into a tensor of shape (N, D).
        Equation (7): x_i = NodeEncoder(T(Base), Role, T(Plugin))
        """
        texts = []
        for agent in agents:
            # 1. Base LLM description (可以参数化，目前写死)
            text_base = "LLM: gpt-3.5-turbo"
            
            # 2. Role description
            text_role = f"Role: {agent.role.name}. {agent.role.description}"
            
            # 3. Tools description
            if agent.tools:
                # 列表转字符串: ['python', 'calc'] -> "python, calc"
                text_tools = f"Tools: {', '.join(agent.tools)}"
            else:
                text_tools = "Tools: None"
            
            # Concatenate with separator
            text = f"{text_base} | {text_role} | {text_tools}"
            texts.append(text)

        # 返回 Tensor (N, 384)
        return self.model.encode(texts, convert_to_tensor=True)

    def encode_query(self, query: str) -> torch.Tensor:
        """
        Encode the task query into a tensor of shape (1, D).
        Used for the virtual task node v_task.
        """
        # 返回 Tensor (1, 384)
        # 注意: encode 默认返回 (D,)，这里我们需要保持维度一致性
        embedding = self.model.encode(query, convert_to_tensor=True)
        return embedding.unsqueeze(0) if embedding.dim() == 1 else embedding
