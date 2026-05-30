import numpy as np
from typing import List, Dict, Tuple
import random

class Agent:
    def __init__(self, agent_id: str, valuation_range: Tuple[float, float]):
        """Agent class representing bidders in auctions
        
        Args:
            agent_id: Unique identifier for the agent
            valuation_range: Range of possible valuations (min, max)
        """
        self.id = agent_id
        self.valuation = random.uniform(valuation_range[0], valuation_range[1])
        self.bid = 0
        self.utility = 0
        
    def set_bid(self, bid: float):
        """Set agent's bid"""
        self.bid = bid
        
    def calculate_utility(self, won: bool, price: float):
        """Calculate agent's utility based on auction outcome
        
        Args:
            won: Whether agent won the auction
            price: Price paid if won (0 if lost)
        """
        if won:
            self.utility = self.valuation - price
        else:
            self.utility = 0

class FirstPriceAuction:
    def __init__(self, agents: List[Agent]):
        """First price sealed bid auction
        
        Args:
            agents: List of participating agents
        """
        self.agents = agents
        self.winner = None
        self.price = 0
        
    def run_auction(self) -> Dict:
        """Run the auction and determine winner/price
        
        Returns:
            Dictionary containing auction results
        """
        # Find highest bidder
        highest_bid = -1
        for agent in self.agents:
            if agent.bid > highest_bid:
                highest_bid = agent.bid
                self.winner = agent
                
        # Winner pays their bid
        if self.winner:
            self.price = self.winner.bid
            
        # Calculate utilities
        for agent in self.agents:
            agent.calculate_utility(agent == self.winner, self.price if agent == self.winner else 0)
            
        return {
            "winner": self.winner.id if self.winner else None,
            "price": self.price,
            "utilities": {a.id: a.utility for a in self.agents}
        }

class SecondPriceAuction:
    def __init__(self, agents: List[Agent]):
        """Second price (Vickrey) auction
        
        Args:
            agents: List of participating agents
        """
        self.agents = agents
        self.winner = None
        self.price = 0
        
    def run_auction(self) -> Dict:
        """Run the auction and determine winner/price
        
        Returns:
            Dictionary containing auction results
        """
        # Sort bids to find highest and second highest
        sorted_agents = sorted(self.agents, key=lambda x: x.bid, reverse=True)
        if len(sorted_agents) >= 1:
            self.winner = sorted_agents[0]
            if len(sorted_agents) >= 2:
                self.price = sorted_agents[1].bid
            else:
                self.price = 0
                
        # Calculate utilities
        for agent in self.agents:
            agent.calculate_utility(agent == self.winner, self.price if agent == self.winner else 0)
            
        return {
            "winner": self.winner.id if self.winner else None, 
            "price": self.price,
            "utilities": {a.id: a.utility for a in self.agents}
        }

def strategic_bidding_simulation():
    """Run simulations comparing first and second price auctions"""
    # Setup agents
    n_agents = 4
    valuation_range = (0, 100)
    agents_first = [Agent(f"Agent_{i}", valuation_range) for i in range(n_agents)]
    agents_second = [Agent(f"Agent_{i}", valuation_range) for i in range(n_agents)]
    
    # In first price auction, bid strategically (shade bid down)
    for agent in agents_first:
        strategic_bid = agent.valuation * (n_agents - 1) / n_agents
        agent.set_bid(strategic_bid)
        
    # In second price auction, bid truthfully
    for agent in agents_second:
        agent.set_bid(agent.valuation)
        
    # Run both auction types
    first_price = FirstPriceAuction(agents_first)
    second_price = SecondPriceAuction(agents_second)
    
    first_results = first_price.run_auction()
    second_results = second_price.run_auction()
    
    print("\nFirst Price Auction Results:")
    print(f"Winner: {first_results['winner']}")
    print(f"Price: {first_results['price']:.2f}")
    print("Utilities:", {k: f"{v:.2f}" for k,v in first_results['utilities'].items()})
    
    print("\nSecond Price Auction Results:")
    print(f"Winner: {second_results['winner']}")
    print(f"Price: {second_results['price']:.2f}")
    print("Utilities:", {k: f"{v:.2f}" for k,v in second_results['utilities'].items()})

# Laboratory Assignments:

#1

"""
1. Basic Mechanism Implementation
- Implement the Agent and FirstPriceAuction classes as shown above
- Run simulations with different numbers of agents and valuation ranges
- Compare theoretical and empirical results

2. Second Price Auction
- Implement the SecondPriceAuction class
- Compare results with first price auction
- Verify that truthful bidding is optimal

3. Strategic Behavior Analysis
- Modify agents to use different bidding strategies
- Study how utilities change with different strategies
- Find optimal bidding functions

4. Revenue Equivalence
- Run many simulations of both auction types
- Calculate average revenue for seller
- Verify revenue equivalence theorem conditions

5. Extensions
- Implement reserve prices
- Add risk averse agents
- Implement other auction formats (Dutch, English)
"""

if __name__ == "__main__":
    strategic_bidding_simulation()


