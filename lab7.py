import random
import statistics
from typing import List, Dict, Tuple, Optional


# =========================
# Agent
# =========================
class Agent:
    def __init__(self, agent_id: str, valuation: float):
        self.id = agent_id
        self.valuation = valuation
        self.bid = 0.0
        self.utility = 0.0

    def set_bid(self, bid: float):
        # bid negative байж болохгүй
        self.bid = max(0.0, bid)

    def calculate_utility(self, won: bool, price: float):
        if won:
            self.utility = self.valuation - price
        else:
            self.utility = 0.0


# =========================
# First Price Auction
# =========================
class FirstPriceAuction:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.winner: Optional[Agent] = None
        self.price = 0.0

    def run_auction(self) -> Dict:
        if not self.agents:
            return {"winner": None, "price": 0.0, "utilities": {}}

        highest_bid = max(agent.bid for agent in self.agents)
        top_agents = [agent for agent in self.agents if agent.bid == highest_bid]

        # тэнцсэн тохиолдолд random winner
        self.winner = random.choice(top_agents)
        self.price = self.winner.bid

        for agent in self.agents:
            agent.calculate_utility(agent == self.winner, self.price if agent == self.winner else 0.0)

        return {
            "winner": self.winner.id,
            "price": self.price,
            "utilities": {a.id: a.utility for a in self.agents},
        }


# =========================
# Second Price Auction
# =========================
class SecondPriceAuction:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.winner: Optional[Agent] = None
        self.price = 0.0

    def run_auction(self) -> Dict:
        if not self.agents:
            return {"winner": None, "price": 0.0, "utilities": {}}

        sorted_agents = sorted(self.agents, key=lambda a: a.bid, reverse=True)
        highest_bid = sorted_agents[0].bid
        top_agents = [agent for agent in self.agents if agent.bid == highest_bid]

        # highest bid тэнцсэн бол winner random
        self.winner = random.choice(top_agents)

        # second price олох
        all_bids = sorted([a.bid for a in self.agents], reverse=True)
        if len(all_bids) >= 2:
            self.price = all_bids[1]
        else:
            self.price = 0.0

        for agent in self.agents:
            agent.calculate_utility(agent == self.winner, self.price if agent == self.winner else 0.0)

        return {
            "winner": self.winner.id,
            "price": self.price,
            "utilities": {a.id: a.utility for a in self.agents},
        }


# =========================
# Helper functions
# =========================
def create_agents_from_valuations(valuations: List[float]) -> List[Agent]:
    return [Agent(f"Agent_{i}", v) for i, v in enumerate(valuations)]


def generate_valuations(n_agents: int, valuation_range: Tuple[float, float]) -> List[float]:
    low, high = valuation_range
    return [random.uniform(low, high) for _ in range(n_agents)]


def apply_strategy(agents: List[Agent], strategy: str):
    n = len(agents)

    for agent in agents:
        v = agent.valuation

        if strategy == "truthful":
            bid = v

        elif strategy == "first_price_equilibrium":
            # Uniform[0,1]-ийн ерөнхий онолын хэлбэрийг [a,b] range дээр scale хийж ашиглаж байна
            # b(v) = a + (n-1)/n * (v-a)
            a = 0.0  # доод хязгаарыг 0 гэж авч үзэж байна
            bid = a + ((n - 1) / n) * (v - a)

        elif strategy == "shade_50":
            bid = 0.5 * v

        elif strategy == "shade_75":
            bid = 0.75 * v

        elif strategy == "overbid_110":
            bid = 1.10 * v

        elif strategy == "random_bid":
            bid = random.uniform(0, v)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        agent.set_bid(bid)


def run_single_comparison(n_agents: int, valuation_range: Tuple[float, float]):
    valuations = generate_valuations(n_agents, valuation_range)

    first_agents = create_agents_from_valuations(valuations)
    second_agents = create_agents_from_valuations(valuations)

    apply_strategy(first_agents, "first_price_equilibrium")
    apply_strategy(second_agents, "truthful")

    first_auction = FirstPriceAuction(first_agents)
    second_auction = SecondPriceAuction(second_agents)

    first_result = first_auction.run_auction()
    second_result = second_auction.run_auction()

    print("=" * 60)
    print(f"n_agents = {n_agents}, valuation_range = {valuation_range}")
    print("Valuations:")
    for i, v in enumerate(valuations):
        print(f"  Agent_{i}: {v:.2f}")

    print("\nFirst Price Auction")
    print(f"  Winner: {first_result['winner']}")
    print(f"  Price : {first_result['price']:.2f}")
    print("  Utilities:")
    for k, v in first_result["utilities"].items():
        print(f"    {k}: {v:.2f}")

    print("\nSecond Price Auction")
    print(f"  Winner: {second_result['winner']}")
    print(f"  Price : {second_result['price']:.2f}")
    print("  Utilities:")
    for k, v in second_result["utilities"].items():
        print(f"    {k}: {v:.2f}")


# =========================
# 1. Basic Mechanism Implementation
# =========================
def experiment_basic_mechanism():
    print("\n" + "#" * 70)
    print("1. BASIC MECHANISM IMPLEMENTATION")
    print("#" * 70)

    test_cases = [
        (3, (0, 100)),
        (4, (0, 100)),
        (5, (20, 80)),
        (6, (50, 150)),
    ]

    for n_agents, valuation_range in test_cases:
        run_single_comparison(n_agents, valuation_range)


# =========================
# 2. Truthful bidding optimality in second-price
# =========================
def test_truthful_bidding_optimality(num_trials: int = 5000, n_agents: int = 4, valuation_range: Tuple[float, float] = (0, 100)):
    print("\n" + "#" * 70)
    print("2. SECOND PRICE AUCTION: TRUTHFUL BIDDING TEST")
    print("#" * 70)

    truthful_utilities = []
    underbid_utilities = []
    overbid_utilities = []

    for _ in range(num_trials):
        valuations = generate_valuations(n_agents, valuation_range)

        # Agent_0 дээр туршилт хийнэ, бусад нь truthful байна
        agents_truth = create_agents_from_valuations(valuations)
        agents_under = create_agents_from_valuations(valuations)
        agents_over = create_agents_from_valuations(valuations)

        # Бусад агентууд truthful
        for agents in [agents_truth, agents_under, agents_over]:
            for i, agent in enumerate(agents):
                if i != 0:
                    agent.set_bid(agent.valuation)

        # Agent_0 өөр өөр strategy хэрэглэнэ
        agents_truth[0].set_bid(agents_truth[0].valuation)              # truthful
        agents_under[0].set_bid(0.8 * agents_under[0].valuation)        # underbid
        agents_over[0].set_bid(1.2 * agents_over[0].valuation)          # overbid

        res_truth = SecondPriceAuction(agents_truth).run_auction()
        res_under = SecondPriceAuction(agents_under).run_auction()
        res_over = SecondPriceAuction(agents_over).run_auction()

        truthful_utilities.append(res_truth["utilities"]["Agent_0"])
        underbid_utilities.append(res_under["utilities"]["Agent_0"])
        overbid_utilities.append(res_over["utilities"]["Agent_0"])

    print(f"Trials: {num_trials}")
    print(f"Average utility of Agent_0 with truthful bid : {statistics.mean(truthful_utilities):.4f}")
    print(f"Average utility of Agent_0 with underbid 0.8v : {statistics.mean(underbid_utilities):.4f}")
    print(f"Average utility of Agent_0 with overbid 1.2v  : {statistics.mean(overbid_utilities):.4f}")

    print("\nConclusion:")
    print("Second-price auction дээр truthful bidding ихэнхдээ хамгийн сайн буюу weakly optimal гарна.")


# =========================
# 3. Strategic behavior analysis
# =========================
def strategic_behavior_analysis(num_trials: int = 5000, n_agents: int = 4, valuation_range: Tuple[float, float] = (0, 100)):
    print("\n" + "#" * 70)
    print("3. STRATEGIC BEHAVIOR ANALYSIS")
    print("#" * 70)

    strategies = [
        "shade_50",
        "shade_75",
        "first_price_equilibrium",
        "truthful",
        "overbid_110",
        "random_bid",
    ]

    avg_utilities = {s: [] for s in strategies}

    for strategy in strategies:
        for _ in range(num_trials):
            valuations = generate_valuations(n_agents, valuation_range)
            agents = create_agents_from_valuations(valuations)

            # Бусад agent-ууд equilibrium strategy хэрэглэнэ
            for i, agent in enumerate(agents):
                if i == 0:
                    continue
                eq_bid = agent.valuation * (n_agents - 1) / n_agents
                agent.set_bid(eq_bid)

            # Agent_0 strategy өөрчилнө
            v0 = agents[0].valuation
            if strategy == "shade_50":
                agents[0].set_bid(0.5 * v0)
            elif strategy == "shade_75":
                agents[0].set_bid(0.75 * v0)
            elif strategy == "first_price_equilibrium":
                agents[0].set_bid(v0 * (n_agents - 1) / n_agents)
            elif strategy == "truthful":
                agents[0].set_bid(v0)
            elif strategy == "overbid_110":
                agents[0].set_bid(1.1 * v0)
            elif strategy == "random_bid":
                agents[0].set_bid(random.uniform(0, v0))

            result = FirstPriceAuction(agents).run_auction()
            avg_utilities[strategy].append(result["utilities"]["Agent_0"])

    print(f"Trials: {num_trials}")
    print(f"n_agents: {n_agents}, valuation_range: {valuation_range}")
    print("\nAverage utility of Agent_0 in First-Price Auction:")
    for s in strategies:
        print(f"  {s:25s}: {statistics.mean(avg_utilities[s]):.4f}")

    best_strategy = max(strategies, key=lambda s: statistics.mean(avg_utilities[s]))
    print(f"\nEmpirically best strategy in this setup: {best_strategy}")


# =========================
# 4. Revenue Equivalence
# =========================
def revenue_equivalence_experiment(num_trials: int = 10000, n_agents: int = 4, valuation_range: Tuple[float, float] = (0, 100)):
    print("\n" + "#" * 70)
    print("4. REVENUE EQUIVALENCE")
    print("#" * 70)

    first_revenues = []
    second_revenues = []

    for _ in range(num_trials):
        valuations = generate_valuations(n_agents, valuation_range)

        first_agents = create_agents_from_valuations(valuations)
        second_agents = create_agents_from_valuations(valuations)

        # First-price equilibrium bidding
        for agent in first_agents:
            bid = agent.valuation * (n_agents - 1) / n_agents
            agent.set_bid(bid)

        # Second-price truthful bidding
        for agent in second_agents:
            agent.set_bid(agent.valuation)

        first_result = FirstPriceAuction(first_agents).run_auction()
        second_result = SecondPriceAuction(second_agents).run_auction()

        first_revenues.append(first_result["price"])
        second_revenues.append(second_result["price"])

    avg_first = statistics.mean(first_revenues)
    avg_second = statistics.mean(second_revenues)

    print(f"Trials: {num_trials}")
    print(f"Average seller revenue (First Price) : {avg_first:.4f}")
    print(f"Average seller revenue (Second Price): {avg_second:.4f}")
    print(f"Absolute difference                  : {abs(avg_first - avg_second):.4f}")

    print("\nInterpretation:")
    print("Risk-neutral bidders, independent private values, symmetric setting зэрэг нөхцөлүүд биелбэл")
    print("хоёр auction-ийн seller revenue ойролцоо байх ёстой.")


# =========================
# Main
# =========================
def main():
    random.seed(42)

    experiment_basic_mechanism()

    test_truthful_bidding_optimality(
        num_trials=5000,
        n_agents=4,
        valuation_range=(0, 100)
    )

    strategic_behavior_analysis(
        num_trials=5000,
        n_agents=4,
        valuation_range=(0, 100)
    )

    revenue_equivalence_experiment(
        num_trials=10000,
        n_agents=4,
        valuation_range=(0, 100)
    )


if __name__ == "__main__":
    main()