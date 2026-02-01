from dataclasses import dataclass

@dataclass
class AgentRole:
    name: str
    description: str

# ============ Domain: Math (GSM8K) ============
MATH_SOLVER = AgentRole(
    name="Math Solver",
    description="You are a math expert. "
        "You will be given a math problem and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n"
        "You will be given some examples you may refer to."
)

MATHEMATICAL_ANALYST = AgentRole(
    name="Mathematical Analyst",
    description="You are a mathematical analyst. "
        "You will be given a math problem, analysis and code from other agents. "
        "You need to first analyze the problem-solving process step by step, where the variables are represented by letters. "
        "Then you substitute the values into the analysis process to perform calculations and get the results."
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n"
        "You will be given some examples you may refer to."
)

MATH_PROGRAMMER = AgentRole(
    name="Programming Expert",
    description="You are a programming expert. "
        "You will be given a math problem, analysis and code from other agents. "
        "Integrate step-by-step reasoning and Python code to solve math problems. "
        "Analyze the question and write functions to solve the problem. "
        "The function should not take any arguments and use the final result as the return value. "
        "The last line of code calls the function you wrote and assigns the return value to the \(answer\) variable. "
        "Use a Python code block to write your response. For example:\n```python\ndef fun():\n x = 10\n y = 20\n return x + y\nanswer = fun()\n```\n"
        "Do not include anything other than Python code blocks in your response."
        "You will be given some examples you may refer to."
)

MATH_INSPECTOR = AgentRole(
    name="Inspector",
    description="You are an Inspector. "
        "You will be given a math problem, analysis and code from other agents. "
        "Check whether the logic/calculation of the problem solving and analysis process is correct(if present). "
        "Check whether the code corresponds to the solution analysis(if present). "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n"
        "You will be given some examples you may refer to."
)

# 2. 也是最重要的：提供一个查找表 (Registry)
ROLE_REGISTRY = {
    "gsm8k":{
        "solver": MATH_SOLVER,
        "analyst": MATH_ANALYST,
        "programmer": MATH_PROGRAMMER,
        "inspector": MATH_INSPECTOR,
    }
}

# 3. 提供一个便捷函数 (Helper Function)
def get_role(domain:str, name: str) -> AgentRole:
    """Safely retrieve a role, maybe with a default fallback."""
    domain_pool = ROLE_REGISTRY.get(domain.lower())
    if not domain_pool:
        raise ValueError(f"Unknown domain: {domain}")
    return domain_pool.get(name.lower())

