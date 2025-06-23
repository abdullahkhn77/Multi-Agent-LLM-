import asyncio
import json
import logging
from dotenv import load_dotenv
import os
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
from dataclasses import dataclass, asdict
import operator

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Pydantic models for structured output
class CodeIssue(BaseModel):
    """Represents a code issue found by an agent"""
    issue: str = Field(description="Description of the issue")
    line: Optional[int] = Field(description="Line number where issue occurs")
    severity: str = Field(description="Severity level: critical, major, minor, suggestion")
    suggestion: Optional[str] = Field(description="Suggested fix for the issue")
    confidence: float = Field(description="Confidence score between 0 and 1")

class AgentResponse(BaseModel):
    """Response from a code review agent"""
    agent_name: str
    category: str
    issues: List[CodeIssue]
    overall_assessment: str
    timestamp: str

# State definition for LangGraph
class ReviewState(TypedDict):
    """State that flows through the review process"""
    code: str
    filename: str
    language: str
    messages: Annotated[List[BaseMessage], operator.add]
    security_review: Optional[AgentResponse]
    performance_review: Optional[AgentResponse]
    style_review: Optional[AgentResponse]
    testing_review: Optional[AgentResponse]
    documentation_review: Optional[AgentResponse]
    agent_communications: Annotated[List[Dict], operator.add]
    final_report: Optional[Dict[str, Any]]
    current_agent: Optional[str]

class SecurityAgent:
    """Security-focused code review agent"""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "SecurityAgent"
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a security expert reviewing code for vulnerabilities.
            Analyze the code for:
            - SQL injection vulnerabilities
            - XSS vulnerabilities  
            - Authentication/authorization issues
            - Input validation problems
            - Cryptographic weaknesses
            - Information disclosure risks
            - Buffer overflows
            - Path traversal vulnerabilities
            
            Be thorough but practical. Focus on real security risks.
            Provide specific line numbers when possible.
            Rate severity as: critical, major, minor, or suggestion."""),
            ("human", "Review this {language} code in file '{filename}':\n\n{code}")
        ])

    async def review(self, state: ReviewState) -> ReviewState:
        """Perform security review of the code"""
        logger.info(f"SecurityAgent reviewing {state['filename']}")
        
        try:
            # Mock response for demonstration - replace with actual LLM call
            mock_issues = [
                CodeIssue(
                    issue="Potential SQL injection vulnerability detected",
                    line=15,
                    severity="critical",
                    suggestion="Use parameterized queries instead of string concatenation",
                    confidence=0.9
                ),
                CodeIssue(
                    issue="User input not properly sanitized",
                    line=8,
                    severity="major",
                    suggestion="Validate and sanitize all user inputs",
                    confidence=0.85
                )
            ]
            
            response = AgentResponse(
                agent_name=self.name,
                category="security",
                issues=mock_issues,
                overall_assessment="Critical security vulnerabilities found that require immediate attention",
                timestamp=datetime.now().isoformat()
            )
            
            state["security_review"] = response
            state["messages"].append(AIMessage(
                content=f"SecurityAgent completed review: {len(mock_issues)} issues found"
            ))
            
            return state
            
        except Exception as e:
            logger.error(f"SecurityAgent error: {e}")
            state["messages"].append(AIMessage(
                content=f"SecurityAgent encountered an error: {str(e)}"
            ))
            return state

class PerformanceAgent:
    """Performance-focused code review agent"""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "PerformanceAgent"
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a performance optimization expert reviewing code.
            Analyze the code for:
            - Time complexity issues (O(n²), O(n³), etc.)
            - Space complexity problems
            - Inefficient algorithms and data structures
            - Memory leaks and resource management
            - Database query optimization opportunities
            - Caching opportunities
            - Unnecessary computations in loops
            - I/O bottlenecks
            
            Focus on measurable performance impacts.
            Provide specific optimization suggestions."""),
            ("human", "Review this {language} code in file '{filename}':\n\n{code}")
        ])

    async def review(self, state: ReviewState) -> ReviewState:
        """Perform performance review of the code"""
        logger.info(f"PerformanceAgent reviewing {state['filename']}")
        
        try:
            mock_issues = [
                CodeIssue(
                    issue="Nested loop creates O(n²) time complexity",
                    line=25,
                    severity="major",
                    suggestion="Consider using a hash map for O(n) lookup instead",
                    confidence=0.88
                ),
                CodeIssue(
                    issue="Database query inside loop causes N+1 problem",
                    line=42,
                    severity="major",
                    suggestion="Move query outside loop or use batch operations",
                    confidence=0.92
                )
            ]
            
            response = AgentResponse(
                agent_name=self.name,
                category="performance",
                issues=mock_issues,
                overall_assessment="Several performance bottlenecks identified that could impact scalability",
                timestamp=datetime.now().isoformat()
            )
            
            state["performance_review"] = response
            state["messages"].append(AIMessage(
                content=f"PerformanceAgent completed review: {len(mock_issues)} issues found"
            ))
            
            return state
            
        except Exception as e:
            logger.error(f"PerformanceAgent error: {e}")
            return state

class StyleAgent:
    """Code style and maintainability agent"""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "StyleAgent"
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a code style and maintainability expert.
            Analyze the code for:
            - Naming convention violations
            - Code organization and structure
            - Design pattern usage
            - Code duplication (DRY principle)
            - Function/class length and complexity
            - Comment quality and placement
            - Consistent formatting
            - SOLID principles adherence
            
            Focus on maintainability and readability improvements."""),
            ("human", "Review this {language} code in file '{filename}':\n\n{code}")
        ])

    async def review(self, state: ReviewState) -> ReviewState:
        """Perform style and maintainability review"""
        logger.info(f"StyleAgent reviewing {state['filename']}")
        
        try:
            mock_issues = [
                CodeIssue(
                    issue="Variable name 'x' is not descriptive",
                    line=5,
                    severity="minor",
                    suggestion="Use descriptive names like 'user_count' or 'total_items'",
                    confidence=0.75
                ),
                CodeIssue(
                    issue="Function is too long (45 lines) and has multiple responsibilities",
                    line=12,
                    severity="major",
                    suggestion="Split into smaller, single-purpose functions",
                    confidence=0.82
                )
            ]
            
            response = AgentResponse(
                agent_name=self.name,
                category="style",
                issues=mock_issues,
                overall_assessment="Code has good structure but could benefit from better naming and organization",
                timestamp=datetime.now().isoformat()
            )
            
            state["style_review"] = response
            state["messages"].append(AIMessage(
                content=f"StyleAgent completed review: {len(mock_issues)} issues found"
            ))
            
            return state
            
        except Exception as e:
            logger.error(f"StyleAgent error: {e}")
            return state

class TestingAgent:
    """Testing and test coverage agent"""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "TestingAgent"
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a testing expert reviewing code for testability and test coverage.
            Analyze the code for:
            - Missing test cases for edge conditions
            - Testability issues (tight coupling, hard dependencies)
            - Test coverage gaps
            - Missing error condition tests
            - Integration test opportunities
            - Mock usage appropriateness
            - Test data management
            - Assertion quality
            
            Focus on improving test coverage and quality."""),
            ("human", "Review this {language} code in file '{filename}':\n\n{code}")
        ])

    async def review(self, state: ReviewState) -> ReviewState:
        """Perform testing review"""
        logger.info(f"TestingAgent reviewing {state['filename']}")
        
        try:
            mock_issues = [
                CodeIssue(
                    issue="No test cases for null/empty input handling",
                    line=1,
                    severity="major",
                    suggestion="Add test cases for null, empty, and boundary value inputs",
                    confidence=0.85
                ),
                CodeIssue(
                    issue="Function has tight coupling making unit testing difficult",
                    line=18,
                    severity="minor",
                    suggestion="Use dependency injection to improve testability",
                    confidence=0.78
                )
            ]
            
            response = AgentResponse(
                agent_name=self.name,
                category="testing",
                issues=mock_issues,
                overall_assessment="Code needs better test coverage, especially for edge cases",
                timestamp=datetime.now().isoformat()
            )
            
            state["testing_review"] = response
            state["messages"].append(AIMessage(
                content=f"TestingAgent completed review: {len(mock_issues)} issues found"
            ))
            
            return state
            
        except Exception as e:
            logger.error(f"TestingAgent error: {e}")
            return state

class DocumentationAgent:
    """Documentation and code clarity agent"""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "DocumentationAgent"
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a documentation expert reviewing code for clarity and documentation.
            Analyze the code for:
            - Missing or inadequate docstrings
            - Unclear or missing comments
            - API documentation completeness
            - Code self-documentation
            - Example usage availability
            - Parameter and return value documentation
            - Complex logic explanation
            - Public interface documentation
            
            Focus on improving code comprehension and usability."""),
            ("human", "Review this {language} code in file '{filename}':\n\n{code}")
        ])

    async def review(self, state: ReviewState) -> ReviewState:
        """Perform documentation review"""
        logger.info(f"DocumentationAgent reviewing {state['filename']}")
        
        try:
            mock_issues = [
                CodeIssue(
                    issue="Function lacks docstring explaining parameters and return value",
                    line=10,
                    severity="minor",
                    suggestion="Add comprehensive docstring with parameter types and return value description",
                    confidence=0.80
                ),
                CodeIssue(
                    issue="Complex algorithm not explained with comments",
                    line=30,
                    severity="minor",
                    suggestion="Add comments explaining the algorithm logic and time complexity",
                    confidence=0.72
                )
            ]
            
            response = AgentResponse(
                agent_name=self.name,
                category="documentation",
                issues=mock_issues,
                overall_assessment="Code would benefit from better documentation and inline comments",
                timestamp=datetime.now().isoformat()
            )
            
            state["documentation_review"] = response
            state["messages"].append(AIMessage(
                content=f"DocumentationAgent completed review: {len(mock_issues)} issues found"
            ))
            
            return state
            
        except Exception as e:
            logger.error(f"DocumentationAgent error: {e}")
            return state

class CoordinatorAgent:
    """Coordinates the multi-agent review process and synthesizes results"""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "CoordinatorAgent"

    async def synthesize_results(self, state: ReviewState) -> ReviewState:
        """Synthesize all agent reviews into a final report"""
        logger.info("CoordinatorAgent synthesizing results")
        
        all_issues = []
        reviews = [
            state.get("security_review"),
            state.get("performance_review"), 
            state.get("style_review"),
            state.get("testing_review"),
            state.get("documentation_review")
        ]
        
        # Collect all issues from all agents
        for review in reviews:
            if review:
                for issue in review.issues:
                    issue_dict = issue.dict()
                    issue_dict["agent_name"] = review.agent_name
                    issue_dict["category"] = review.category
                    all_issues.append(issue_dict)
        
        # Prioritize issues by severity and confidence
        severity_weights = {"critical": 4, "major": 3, "minor": 2, "suggestion": 1}
        all_issues.sort(
            key=lambda x: severity_weights.get(x["severity"], 1) * x["confidence"], 
            reverse=True
        )
        
        # Generate summary statistics
        total_issues = len(all_issues)
        critical_count = len([i for i in all_issues if i["severity"] == "critical"])
        major_count = len([i for i in all_issues if i["severity"] == "major"])
        
        # Group issues by category
        issues_by_category = {}
        for issue in all_issues:
            category = issue["category"]
            if category not in issues_by_category:
                issues_by_category[category] = []
            issues_by_category[category].append(issue)
        
        # Generate overall recommendation
        if critical_count > 0:
            recommendation = f"REQUIRES IMMEDIATE ATTENTION: {critical_count} critical issues found"
        elif major_count > 3:
            recommendation = f"SIGNIFICANT IMPROVEMENTS NEEDED: {major_count} major issues identified"
        elif major_count > 0:
            recommendation = f"IMPROVEMENTS RECOMMENDED: {major_count} major issues found"
        else:
            recommendation = "CODE QUALITY GOOD: Only minor suggestions provided"
        
        # Create final report
        final_report = {
            "summary": {
                "total_issues": total_issues,
                "critical_issues": critical_count,
                "major_issues": major_count,
                "recommendation": recommendation,
                "review_timestamp": datetime.now().isoformat()
            },
            "issues_by_category": issues_by_category,
            "all_issues": all_issues[:20],  # Top 20 issues
            "agent_assessments": {
                review.agent_name: review.overall_assessment 
                for review in reviews if review
            }
        }
        
        state["final_report"] = final_report
        state["messages"].append(AIMessage(
            content=f"Review completed: {total_issues} total issues found across all categories"
        ))
        
        return state

    async def detect_conflicts(self, state: ReviewState) -> ReviewState:
        """Detect and resolve conflicts between agent assessments"""
        logger.info("CoordinatorAgent detecting conflicts")
        
        # Simple conflict detection - agents disagreeing on same line
        line_issues = {}
        reviews = [
            state.get("security_review"),
            state.get("performance_review"), 
            state.get("style_review"),
            state.get("testing_review"),
            state.get("documentation_review")
        ]
        
        for review in reviews:
            if review:
                for issue in review.issues:
                    if issue.line:
                        if issue.line not in line_issues:
                            line_issues[issue.line] = []
                        line_issues[issue.line].append({
                            "agent": review.agent_name,
                            "severity": issue.severity,
                            "issue": issue.issue
                        })
        
        # Find conflicts (different severities on same line)
        conflicts = []
        for line_num, issues in line_issues.items():
            if len(issues) > 1:
                severities = [issue["severity"] for issue in issues]
                if len(set(severities)) > 1:
                    conflicts.append({
                        "line": line_num,
                        "conflicting_assessments": issues,
                        "resolution": "Multiple agents identified different severity levels for the same line"
                    })
        
        if conflicts:
            state["agent_communications"].extend([{
                "type": "conflict_detected",
                "conflicts": conflicts,
                "timestamp": datetime.now().isoformat()
            }])
        
        return state

def create_review_graph():
    """Create the LangGraph workflow for multi-agent code review"""
    
    # Initialize LLM (mock for this example)
    llm = ChatOpenAI(model="gpt-4", temperature=0.1)
    
    # Initialize agents
    security_agent = SecurityAgent(llm)
    performance_agent = PerformanceAgent(llm)
    style_agent = StyleAgent(llm)
    testing_agent = TestingAgent(llm)
    documentation_agent = DocumentationAgent(llm)
    coordinator_agent = CoordinatorAgent(llm)
    
    # Create the graph
    workflow = StateGraph(ReviewState)
    
    # Add nodes for each agent
    workflow.add_node("security_review", security_agent.review)
    workflow.add_node("performance_review", performance_agent.review)
    workflow.add_node("style_review", style_agent.review)
    workflow.add_node("testing_review", testing_agent.review)
    workflow.add_node("documentation_review", documentation_agent.review)
    workflow.add_node("detect_conflicts", coordinator_agent.detect_conflicts)
    workflow.add_node("synthesize_results", coordinator_agent.synthesize_results)
    
    # Set entry point
    workflow.set_entry_point("security_review")
    
    # Add edges to create the execution flow
    workflow.add_edge("security_review", "performance_review")
    workflow.add_edge("performance_review", "style_review")
    workflow.add_edge("style_review", "testing_review")
    workflow.add_edge("testing_review", "documentation_review")
    workflow.add_edge("documentation_review", "detect_conflicts")
    workflow.add_edge("detect_conflicts", "synthesize_results")
    workflow.add_edge("synthesize_results", END)
    
    return workflow.compile()

class MultiAgentCodeReviewSystem:
    """Main system that orchestrates the multi-agent code review"""
    
    def __init__(self):
        self.graph = create_review_graph()
        logger.info("Multi-Agent Code Review System with LangGraph initialized")

    def detect_language(self, filename: str) -> str:
        """Detect programming language from filename"""
        extension_map = {
            ".py": "Python",
            ".js": "JavaScript", 
            ".java": "Java",
            ".cpp": "C++",
            ".c": "C",
            ".cs": "C#",
            ".rb": "Ruby",
            ".php": "PHP",
            ".go": "Go",
            ".rs": "Rust",
            ".ts": "TypeScript"
        }
        
        for ext, lang in extension_map.items():
            if filename.endswith(ext):
                return lang
        return "Unknown"

    async def review_code(self, code: str, filename: str) -> Dict[str, Any]:
        """Execute the multi-agent code review workflow"""
        logger.info(f"Starting review for {filename}")
        
        # Initialize state
        initial_state = {
            "code": code,
            "filename": filename, 
            "language": self.detect_language(filename),
            "messages": [HumanMessage(content=f"Please review the code in {filename}")],
            "security_review": None,
            "performance_review": None,
            "style_review": None,
            "testing_review": None,
            "documentation_review": None,
            "agent_communications": [],
            "final_report": None,
            "current_agent": None
        }
        
        # Execute the workflow
        try:
            result = await self.graph.ainvoke(initial_state)
            return result["final_report"]
        except Exception as e:
            logger.error(f"Review workflow failed: {e}")
            return {
                "error": str(e),
                "summary": {
                    "total_issues": 0,
                    "critical_issues": 0,
                    "recommendation": "Review failed due to system error"
                }
            }
