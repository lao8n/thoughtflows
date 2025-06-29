import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from pydantic import BaseModel, Field
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from autogen_core.models import LLMMessage
from autogen_agentchat.messages import TextMessage
import numpy as np
import re

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
    def __init__(self, model_name: str = "gpt-4o-mini"):
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found. Please check your .env file.")
            
        self.model_client = OpenAIChatCompletionClient(model=model_name, api_key=api_key)
        
        # New agent for current situation
        self.current_situation_describer = AssistantAgent(
            "CurrentSituationDescriber",
            model_client=self.model_client,
            system_message="""You are an expert at summarizing current events.\nFor the given prompt, write a brief historical-style note as if this is a historical event.\nProvide:\n1. Name of the event/situation (e.g., 'US Debt Crisis 2025')\n2. Brief description (2-3 sentences)""",
            output_content_type=HistoricalExample
        )

        # Initialize the historical finder
        self.historical_finder = AssistantAgent(
            "HistoricalFinder",
            model_client=self.model_client,
            system_message="""You are an expert at finding relevant historical examples.
            For any given prompt, find 3-5 historical examples that are most similar.
            For each example, provide:
            1. Name of the event/situation
            2. Brief description (2-3 sentences)""",
            output_content_type=HistoricalExamplesList
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
            output_content_type=NearestMatch
        )

    def _create_similarity_analyzer(self, example: HistoricalExample, index: int) -> AssistantAgent:
        """Create a similarity analyzer agent for a specific example."""
        # Sanitize the example name to be a valid Python identifier
        base_name = re.sub(r'\W|^(?=\d)', '_', example.name)
        agent_name = f"SA_{base_name}"
        # Truncate to ensure <64 chars
        agent_name = agent_name[:60]
        return AssistantAgent(
            agent_name,
            model_client=self.model_client,
            system_message=f"""You are an expert at analyzing similarities and differences between situations.
            Analyze this specific historical example:
            Name: {example.name}
            Description: {example.description}
            
            For this example, analyze:
            1. Key similarities with the current situation
            2. Important differences""",
            output_content_type=SimilarityAnalysis
        )

    async def run_analysis(self, prompt: str) -> Dict[str, Any]:
        # Sanitize prompt for folder name
        safe_prompt = re.sub(r'\W+', '_', prompt.strip())
        # Get project root (parent of src)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "flows", safe_prompt)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "output.txt")
        flow_graph_file = os.path.join(output_dir, "flow_graph.png")
        # Step 1: Generate a HistoricalExample for the current prompt
        current_example_result = await self.current_situation_describer.on_messages(
            messages=[TextMessage(source="user", content=prompt)],
            cancellation_token=None
        )
        if hasattr(current_example_result, 'chat_message') and hasattr(current_example_result.chat_message, 'content'):
            current_example = current_example_result.chat_message.content
        else:
            current_example = current_example_result
        # Step 2: Get the historical examples as before
        historical_finder_prompt = f"{current_example.name}: {current_example.description}"
        examples_result = await self.historical_finder.on_messages(
            messages=[TextMessage(source="user", content=historical_finder_prompt)],
            cancellation_token=None
        )
        print(f"examples_result type: {type(examples_result)}")
        print(f"examples_result content: {examples_result}")
        # Extract the actual content from the Response object
        if hasattr(examples_result, 'chat_message') and hasattr(examples_result.chat_message, 'content'):
            examples_data = examples_result.chat_message.content
        else:
            examples_data = examples_result
        print(f"examples_data type: {type(examples_data)}")
        print(f"examples_data content: {examples_data}")
        if not isinstance(examples_data, HistoricalExamplesList):
            print("examples_data is not a HistoricalExamplesList")
            raise ValueError("Failed to get historical examples")
        if not examples_data.examples:
            print("examples_data.examples is empty")
            raise ValueError("No historical examples found")
        print(f"Found {len(examples_data.examples)} historical examples")
        # Step 3: Prepend the current example
        examples_data.examples = [current_example] + examples_data.examples
        # Write historical examples to output file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Prompt: {prompt}\n\n")
            f.write("--- Historical Examples ---\n")
            for ex in examples_data.examples:
                f.write(f"Name: {ex.name}\nDescription: {ex.description}\n\n")
        # Create one similarity analyzer per example
        self.similarity_analyzers = [
            self._create_similarity_analyzer(example, i)
            for i, example in enumerate(examples_data.examples)
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
        # Visualize the graph in the output directory
        self.visualize_graph(graph, filename=flow_graph_file)
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
        # Process results and write to files
        final_result = self._process_results(results)
        # Write similarity analyses and best match to output_file
        if "similarity_analysis" in final_result:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write("\n--- Similarity Analyses ---\n")
                for analysis in final_result["similarity_analysis"]:
                    f.write(f"Example: {analysis.example_name}\nSimilarities: {analysis.similarities}\nDifferences: {analysis.differences}\n\n")
        if "nearest_match" in final_result:
            with open(output_file, "a", encoding="utf-8") as f:
                match = final_result["nearest_match"]
                f.write("\n--- Best Match ---\n")
                f.write(f"Best Match: {match.best_match}\nReasoning: {match.reasoning}\nPredictions: {match.predictions}\n")
        return final_result
    
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

    def visualize_graph(self, graph, filename: str = "flow_graph.png"):
        """Visualize the flow graph using NetworkX and Matplotlib."""
        print(f"Graph type: {type(graph)}")
        plt.figure(figsize=(12, 8))
        # Get nodes from the custom graph object
        if hasattr(graph, 'nodes'):
            nodes = list(graph.nodes.keys())
            node_objs = graph.nodes
            print(f"Nodes: {nodes}")
        else:
            print("Cannot visualize graph - no nodes attribute")
            plt.close()
            return
        # Simple circular layout
        n_nodes = len(nodes)
        angles = [2 * np.pi * i / n_nodes for i in range(n_nodes)]
        x_coords = [np.cos(angle) for angle in angles]
        y_coords = [np.sin(angle) for angle in angles]
        node_pos = {node: (x, y) for node, x, y in zip(nodes, x_coords, y_coords)}
        # Plot nodes
        plt.scatter(x_coords, y_coords, c='lightblue', s=1000, alpha=0.6)
        # Add node labels
        for node in nodes:
            x, y = node_pos[node]
            plt.annotate(str(node), (x, y), ha='center', va='center', fontsize=10, fontweight='bold')
        # Draw edges
        for node in nodes:
            node_obj = node_objs[node]
            if hasattr(node_obj, 'edges'):
                for edge in node_obj.edges:
                    target = edge.target
                    if target in node_pos:
                        x0, y0 = node_pos[node]
                        x1, y1 = node_pos[target]
                        plt.arrow(x0, y0, x1-x0, y1-y0, length_includes_head=True, head_width=0.05, head_length=0.1, fc='gray', ec='gray', alpha=0.5)
        plt.title("Nearest Neighbor Flow Graph")
        plt.axis('equal')
        plt.axis('off')
        plt.savefig(filename)
        plt.close()
        print(f"Graph saved to {filename}")

def get_output_dir(prompt):
    safe_prompt = re.sub(r'\W+', '_', prompt.strip())
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
    output_dir = os.path.join(base_dir, "flows", safe_prompt)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

async def main():
    # Example usage
    flow = NearestNeighborFlow()
    result = await flow.run_analysis("US debt crisis")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
