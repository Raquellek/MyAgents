# shalgalt2.ipynb

Source notebook: [`shalgalt2.ipynb`](../shalgalt2.ipynb)

## Cell 1

```python
from magent2.environments import battle_v4

env = battle_v4.parallel_env(
    map_size=16,
    max_cycles=25,
    render_mode=None
)

obs = env.reset(seed=42)
infos = {}

print("Нийт агент:", len(env.agents))
print(env.agents[:10])

env.close()
```

## Cell 2

```python
from itertools import combinations
import random
from magent2.environments import battle_v4


# -----------------------------
# 1. 81-с дээш агенттай battle_v4 орчин үүсгэх
# -----------------------------
def make_battle_env_for_81_agents():
    for map_size in [20, 25, 30, 35, 40, 45, 50]:
        try:
            env = battle_v4.parallel_env(
                map_size=map_size,
                max_cycles=25,
                render_mode=None
            )
        except TypeError:
            env = battle_v4.parallel_env(
                map_size=map_size,
                max_cycles=25
            )

        env.reset(seed=42)
        agents = list(env.agents)

        if len(agents) >= 81:
            return env, agents, map_size

        env.close()

    raise ValueError("81-с дээш агенттай battle_v4 орчин үүсгэж чадсангүй.")


# -----------------------------
# 2. Агентын нэрээс баг ба дугаар авах
# Жишээ: red_0 -> team=red, number=0
# -----------------------------
def get_agent_info(agent_name):
    team, number = agent_name.rsplit("_", 1)
    return team, int(number)


def print_agent(agent_name):
    team, number = get_agent_info(agent_name)
    return f"{agent_name} | баг={team}, дугаар={number}"


# -----------------------------
# 3. Орчин үүсгэх
# -----------------------------
env, agents, used_map_size = make_battle_env_for_81_agents()

print("battle_v4 орчин амжилттай үүсэв")
print("map_size:", used_map_size)
print("Нийт агентын тоо:", len(agents))


# -----------------------------
# 4. Зөвхөн 81 агент сонгох
# -----------------------------
selected_agents = agents[:81]

print("\n==============================")
print("Сонгогдсон 81 агент")
print("==============================")

for agent in selected_agents:
    print("  -", print_agent(agent))


# -----------------------------
# 5. 81 агентыг эвсэл болгон хуваах
# -----------------------------
def create_coalitions(agent_list, coalition_size=9, shuffle=True, seed=42):
    agents_copy = agent_list.copy()

    if shuffle:
        random.seed(seed)
        random.shuffle(agents_copy)

    coalitions = []

    for i in range(0, len(agents_copy), coalition_size):
        coalition = agents_copy[i:i + coalition_size]
        coalitions.append(coalition)

    return coalitions


coalition_size = 9
coalitions = create_coalitions(
    selected_agents,
    coalition_size=coalition_size,
    shuffle=True,
    seed=42
)


print("\n==============================")
print("81 агентаар байгуулсан эвсэлүүд")
print("==============================")
print("Coalition size:", coalition_size)
print("Нийт эвслийн тоо:", len(coalitions))


for idx, coalition in enumerate(coalitions, start=1):
    print(f"\nЭвсэл {idx} | хэмжээ = {len(coalition)}")

    for agent in coalition:
        team, number = get_agent_info(agent)
        print(f"  - agent name: {agent}, баг: {team}, дугаар: {number}")


env.close()
```

