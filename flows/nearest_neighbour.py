import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from autogen import AssistantAgent, OpenAIWrapper
from autogen.agentchat.contrib.text_analyzer_agent import TextAnalyzerAgent
from autogen.agentchat.contrib.graph_flow import DiGraphBuilder, GraphFlow
from autogen_agentchat.conditions import MaxMessageTermination
from pydantic import BaseModel, Field
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Pydantic models for structured outputs
class HistoricalExample(BaseModel):
    name: str = Field(..., description="Name of the historical event/situation")
    description: str = Field(..., description="Brief description of the event/situation")

class HistoricalExamplesList(BaseModel):
    examples: List[HistoricalExample] = Field(..., description="List of historical examples")

class SimilarityAnalysis(BaseModel):
    example_name: str = Field(..., description="Name of the example being analyzed")
    similarities: List[str] = Field(..., description="List of key similarities with the current situation")
    differences: List[str] = Field(..., description="List of important differences from the current situation")

class SimilarityAnalysesList(BaseModel):
    analyses: List[SimilarityAnalysis] = Field(..., description="List of similarity analyses")

class NearestMatch(BaseModel):
    best_match: str = Field(..., description="Name of the most similar historical example")
    reasoning: str = Field(..., description="Explanation of why this is the best match")
    predictions: List[str] = Field(..., description="List of predictions based on this example")

class NearestNeighborFlow:
    def __init__(self, model_name: str = "gpt-4"):
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found. Please check your .env file.")
            
        self.model_client = OpenAIWrapper(model=model_name, api_key=api_key)
        
        # Initialize the historical finder
        self.historical_finder = AssistantAgent(
            "HistoricalFinder",
            model_client=self.model_client,
            system_message="""You are an expert at finding relevant historical examples.
            For any given prompt, find 3-5 historical examples that are most similar.
            For each example, provide:
            1. Name of the event/situation
            2. Brief description (2-3 sentences)""",
            response_format=HistoricalExamplesList
        )
        
        # Initialize the nearest neighbor
        self.nearest_neighbor = AssistantAgent(
            "NearestNeighbor",
            model_client=self.model_client,
            system_message="""You are an expert at finding the most relevant historical precedent.
            Analyze all examples and their similarities/differences to determine:
            1. Which example is most similar to the current situation
            2. Why it's the best match
            3. What predictions can be made based on this example""",
            response_format=NearestMatch
        )

    def _create_similarity_analyzer(self, example: HistoricalExample, index: int) -> AssistantAgent:
        """Create a similarity analyzer agent for a specific example."""
        return AssistantAgent(
            f"SimilarityAnalyzer_{index}",
            model_client=self.model_client,
            system_message=f"""You are an expert at analyzing similarities and differences between situations.
            Analyze this specific historical example:
            Name: {example.name}
            Description: {example.description}
            
            For this example, analyze:
            1. Key similarities with the current situation
            2. Important differences""",
            response_format=SimilarityAnalysis
        )

    async def run_analysis(self, prompt: str) -> Dict[str, Any]:
        # First, get the historical examples
        examples_result = await self.historical_finder.generate_reply(
            sender=None, 
            messages=[{"role": "user", "content": prompt}]
        )
        
        if not isinstance(examples_result, HistoricalExamplesList):
            raise ValueError("Failed to get historical examples")
            
        if not examples_result.examples:
            raise ValueError("No historical examples found")
            
        print(f"Found {len(examples_result.examples)} historical examples")
        
        # Create one similarity analyzer per example
        self.similarity_analyzers = [
            self._create_similarity_analyzer(example, i)
            for i, example in enumerate(examples_result.examples)
        ]
        
        # Create the flow graph
        builder = DiGraphBuilder()
        
        # Add nodes
        builder.add_node(self.historical_finder)
        for analyzer in self.similarity_analyzers:
            builder.add_node(analyzer)
        builder.add_node(self.nearest_neighbor)
        
        # Create fan-out edges from historical_finder to each similarity analyzer
        for analyzer in self.similarity_analyzers:
            builder.add_edge(self.historical_finder, analyzer)
        
        # Create edges from all similarity analyzers to nearest_neighbor
        for analyzer in self.similarity_analyzers:
            builder.add_edge(analyzer, self.nearest_neighbor)
        
        graph = builder.build()
        
        # Visualize the graph
        self.visualize_graph(graph)
        
        # Create the team
        team = GraphFlow(
            participants=[self.historical_finder] + self.similarity_analyzers + [self.nearest_neighbor],
            graph=graph,
            termination_condition=MaxMessageTermination(20)
        )
        
        # Run the analysis
        results = []
        async for event in team.run_stream(task=prompt):
            results.append(event)
            
        return self._process_results(results)
    
    def _process_results(self, events: List[Any]) -> Dict[str, Any]:
        final_result = {}
        similarity_analyses = []
        
        for event in events:
            if hasattr(event, 'content'):
                try:
                    if isinstance(event.content, HistoricalExamplesList):
                        final_result["historical_examples"] = event.content.examples
                    elif isinstance(event.content, SimilarityAnalysis):
                        similarity_analyses.append(event.content)
                    elif isinstance(event.content, NearestMatch):
                        final_result["nearest_match"] = event.content
                except Exception as e:
                    print(f"Error processing result: {e}")
                    continue
        
        # Add all similarity analyses to the result
        if similarity_analyses:
            final_result["similarity_analysis"] = similarity_analyses
            
        return final_result

    def visualize_graph(self, graph: nx.DiGraph, filename: str = "flow_graph.png"):
        """Visualize the flow graph using NetworkX and Matplotlib."""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue', 
                             node_size=2000, alpha=0.6)
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos, edge_color='gray', 
                             arrows=True, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold')
        
        plt.title("Nearest Neighbor Flow Graph")
        plt.axis('off')
        plt.savefig(filename)
        plt.close()

async def main():
    # Example usage
    flow = NearestNeighborFlow()
    result = await flow.run_analysis("US debt crisis")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
