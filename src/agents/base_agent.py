from dataclasses import dataclass

@dataclass
class AgentMessage:
    role: str
    content: str

class BaseAgent:
    def __init__(self, role, tools, llm_client, max_react_steps=5):
        self.role = role
        self.tools = tools
        self.llm_client = llm_client
        self.max_react_steps = max_react_steps
        self.system_prompt = self._construct_system_prompt()

    def _aggregate_messages(self, query, upstream_msgs):
        prompt = f"Origin Task: {query}\n"

        if upstream_msgs:
            prompt += "Upstream Messages:\n"
            for msg in upstream_msgs:
                prompt += f"- message from{msg.role}: {msg.content}\n"

        return prompt

    def _construct_system_prompt(self,):
        prompt = f"You are a helpless assistant called {self.role.name}.\n"
        prompt += f"Description: {self.role.description}\n"

        if self.tools:
            prompt += f"You have access to the following tools: {self.tools}\n"
        
        return prompt

    def step(self, query, upstream_msgs):
        # 聚合来自上游智能体的消息
        # 组装prompt
        # 调用llm解决问题
        # 解析并执行工具
        # 返回最后的回复
        user_content = self._aggregate_messages(query, upstream_msgs)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
        for _ in range(self.max_react_steps):
            response = self.llm_client.chat(messages)
            if response.is_tool_call:
                tool_result = self._execute_tool(response.tool_call)
                messages.append(response.message)
                messages.append(tool_result)
            else:
                return AgentMessage(role=self.role.name, content=response.content)
        return AgentMessage(role=self.role.name, content="Error: MAX steps reached")
