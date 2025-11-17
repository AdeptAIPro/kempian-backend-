"""
Graph Neural Network for Career Trajectory Analysis
Models career progression as graphs and predicts behavioral patterns
"""

from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np

# Try to import torch_geometric, fallback if not available
try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not available. GNN features will be disabled.")
    # Create dummy classes for fallback
    class GCNConv:
        def __init__(self, *args, **kwargs):
            pass
        def forward(self, *args, **kwargs):
            return torch.randn(1, 1)
    
    class Data:
        def __init__(self, *args, **kwargs):
            pass
    
    def global_mean_pool(x, batch):
        return torch.mean(x, dim=0, keepdim=True)


@dataclass
class CareerNode:
    """Represents a career milestone/role in the career graph"""
    role_title: str
    company: str
    duration_months: int
    skills: List[str]
    achievements: List[str]
    level: str  # junior, mid, senior, lead, executive
    domain: str


class CareerGraphGNN(nn.Module):
    """
    Graph Neural Network for career trajectory analysis
    Models career progression as a graph with roles as nodes and transitions as edges
    """
    
    def __init__(self, node_features: int = 768, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            print("Warning: GNN disabled due to missing torch_geometric")
            self.disabled = True
            return
        
        self.disabled = False
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
        # Behavioral prediction heads
        self.leadership_head = nn.Linear(output_dim, 1)
        self.innovation_head = nn.Linear(output_dim, 1)
        self.growth_head = nn.Linear(output_dim, 1)
        
    def forward(self, x, edge_index, batch=None):
        if self.disabled:
            # Return dummy predictions when GNN is disabled
            return {
                'embeddings': torch.randn(1, 128),
                'leadership': torch.tensor([[0.5]]),
                'innovation': torch.tensor([[0.5]]), 
                'growth': torch.tensor([[0.5]])
            }
        
        # Graph convolutions
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        
        # Global pooling for graph-level representation
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Behavioral predictions
        leadership = torch.sigmoid(self.leadership_head(x))
        innovation = torch.sigmoid(self.innovation_head(x))
        growth = torch.sigmoid(self.growth_head(x))
        
        return {
            'embeddings': x,
            'leadership': leadership,
            'innovation': innovation, 
            'growth': growth
        }


class CareerGraphBuilder:
    """
    Builds career trajectory graph from resume entities
    """
    
    def __init__(self, bert_encoder):
        self.bert_encoder = bert_encoder
        
    def parse_career_timeline(self, entities: Dict, resume_text: str) -> List[CareerNode]:
        """Extract structured career nodes from entities"""
        roles = entities.get("ROLES", [])
        companies = entities.get("COMPANIES", [])
        dates = entities.get("DATES", [])
        skills = entities.get("SKILLS", [])
        
        # Simple heuristic to match roles with companies and dates
        nodes = []
        for i, role in enumerate(roles):
            company = companies[i] if i < len(companies) else "Unknown"
            
            # Extract skills mentioned near this role (basic proximity)
            role_skills = [skill for skill in skills if skill.lower() in resume_text.lower()]
            
            node = CareerNode(
                role_title=role,
                company=company,
                duration_months=12,  # Default, could be parsed from dates
                skills=role_skills[:5],  # Top 5 skills
                achievements=[],  # Could be extracted with more sophisticated NLP
                level=self._infer_seniority(role),
                domain=self._infer_domain(role, role_skills)
            )
            nodes.append(node)
        
        return nodes
    
    def _infer_seniority(self, role_title: str) -> str:
        """Infer seniority level from role title"""
        title_lower = role_title.lower()
        
        if any(word in title_lower for word in ['ceo', 'cto', 'vp', 'director', 'head']):
            return 'executive'
        elif any(word in title_lower for word in ['lead', 'principal', 'staff', 'architect']):
            return 'lead'
        elif any(word in title_lower for word in ['senior', 'sr']):
            return 'senior'
        elif any(word in title_lower for word in ['junior', 'jr', 'intern', 'associate']):
            return 'junior'
        else:
            return 'mid'
    
    def _infer_domain(self, role_title: str, skills: List[str]) -> str:
        """Infer domain from role and skills"""
        combined_text = f"{role_title} {' '.join(skills)}"
        
        # Simple domain inference (could use the DomainSpecificBERT here)
        text_lower = combined_text.lower()
        
        if any(word in text_lower for word in ['medical', 'clinical', 'healthcare']):
            return 'healthcare'
        elif any(word in text_lower for word in ['research', 'academic', 'phd']):
            return 'research'
        elif any(word in text_lower for word in ['software', 'programming', 'development']):
            return 'tech'
        elif any(word in text_lower for word in ['finance', 'banking', 'investment']):
            return 'finance'
        elif any(word in text_lower for word in ['legal', 'law', 'attorney']):
            return 'legal'
        else:
            return 'general'
    
    def build_graph(self, career_nodes: List[CareerNode], bert_encoder=None) -> Data:
        """Build PyTorch Geometric graph from career nodes"""
        
        if not career_nodes:
            # Return empty graph
            return Data(x=torch.zeros((1, 768)), edge_index=torch.zeros((2, 0), dtype=torch.long))
        
        # Encode nodes using BERT (simplified for this version)
        node_features = []
        for node in career_nodes:
            # Create a simple text representation
            node_text = f"{node.role_title} {node.company} {' '.join(node.skills)}"
            
            # For this example, create random embeddings (in real implementation, use BERT)
            if bert_encoder and hasattr(bert_encoder, 'encode'):
                try:
                    from .domain_bert import DomainType
                    domain = DomainType(node.domain) if node.domain in ['healthcare', 'research', 'tech', 'finance', 'legal'] else DomainType.GENERAL
                    embedding = bert_encoder.encode(node_text, domain).squeeze()
                except:
                    # Fallback to random embedding
                    embedding = torch.randn(768)
            else:
                # Fallback to random embedding for demo
                embedding = torch.randn(768)
                
            node_features.append(embedding)
        
        x = torch.stack(node_features)
        
        # Create edges (sequential career progression)
        edge_index = []
        for i in range(len(career_nodes) - 1):
            edge_index.append([i, i + 1])  # Forward progression
            edge_index.append([i + 1, i])  # Backward connection
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)


class PretrainedCareerGNN:
    """
    Wrapper for a pretrained career GNN model
    In practice, this would load weights from a trained model
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = CareerGraphGNN()
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.model.eval()
            except Exception as e:
                print(f"Failed to load pretrained GNN model: {e}")
                print("Using randomly initialized model.")
        
        self.graph_builder = None  # Will be set when needed
    
    def set_graph_builder(self, graph_builder: CareerGraphBuilder):
        """Set the graph builder instance"""
        self.graph_builder = graph_builder
    
    def predict_career_trajectory(self, entities: Dict, resume_text: str, 
                                bert_encoder=None) -> Dict[str, float]:
        """
        Predict behavioral patterns from career trajectory
        """
        if not self.graph_builder:
            self.graph_builder = CareerGraphBuilder(bert_encoder)
        
        # Build career graph
        career_nodes = self.graph_builder.parse_career_timeline(entities, resume_text)
        career_graph = self.graph_builder.build_graph(career_nodes, bert_encoder)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(career_graph.x, career_graph.edge_index)
        
        return {
            'leadership': float(predictions['leadership'].item()),
            'innovation': float(predictions['innovation'].item()),
            'growth_potential': float(predictions['growth'].item()),
            'career_complexity': len(career_nodes),
            'domain_diversity': len(set(node.domain for node in career_nodes)) if career_nodes else 0
        }
    
    def analyze_career_patterns(self, career_nodes: List[CareerNode]) -> Dict[str, any]:
        """
        Analyze career patterns and progression
        """
        if not career_nodes:
            return {
                'progression_score': 0.0,
                'stability_score': 0.5,
                'breadth_score': 0.0,
                'insights': ['Insufficient career history']
            }
        
        # Analyze seniority progression
        seniority_levels = ['junior', 'mid', 'senior', 'lead', 'executive']
        level_progression = [node.level for node in career_nodes]
        
        progression_score = 0.0
        if len(level_progression) > 1:
            level_indices = [seniority_levels.index(level) if level in seniority_levels else 1 
                           for level in level_progression]
            # Check if generally progressing upward
            progression_score = max(0.0, (level_indices[-1] - level_indices[0]) / 4.0)
        
        # Calculate stability (average tenure)
        avg_tenure = sum(node.duration_months for node in career_nodes) / len(career_nodes)
        stability_score = min(1.0, avg_tenure / 24.0)  # 2 years = 1.0
        
        # Calculate breadth (domain diversity)
        unique_domains = set(node.domain for node in career_nodes)
        breadth_score = min(1.0, len(unique_domains) / 3.0)  # 3+ domains = 1.0
        
        # Generate insights
        insights = []
        if progression_score > 0.7:
            insights.append("Strong upward career progression")
        elif progression_score < 0.3:
            insights.append("Limited seniority advancement")
        
        if stability_score > 0.7:
            insights.append("Demonstrates job stability")
        elif stability_score < 0.3:
            insights.append("Frequent job changes may indicate instability")
        
        if breadth_score > 0.6:
            insights.append("Diverse cross-domain experience")
        
        return {
            'progression_score': progression_score,
            'stability_score': stability_score,
            'breadth_score': breadth_score,
            'insights': insights,
            'career_length': len(career_nodes),
            'domains_covered': list(unique_domains)
        }