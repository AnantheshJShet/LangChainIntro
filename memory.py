"""
LangGraph Memory Concepts - Educational Reference
================================================

This file demonstrates different memory concepts in LangGraph, from basic to advanced.
Each section builds upon the previous one, showing progressive complexity.

Table of Contents:
1. Conversation Memory - Chat History
2. Structured Memory - Typed State
3. Persistent Memory - Database Storage
4. Vector Memory - Semantic Search
5. Multi-Agent Memory - Shared State
6. Advanced Memory - Custom Memory Classes
"""

import asyncio
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
import json
import os

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

# ============================================================================
# 1. CONVERSATION MEMORY - Chat History
# ============================================================================

def conversation_memory_example():
    """
    Conversation memory example showing how to maintain chat history.
    This demonstrates storing and retrieving conversation context.
    """
    print("=== 1. CONVERSATION MEMORY - Chat History ===")
    
    class ConversationState(TypedDict):
        messages: List[Dict[str, str]]
        user_input: str
        response: str
    
    def process_user_input(state: ConversationState) -> ConversationState:
        """Process user input and add to conversation history."""
        messages = state.get("messages", [])
        user_input = state["user_input"]
        
        # Add user message to history
        messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        return {"messages": messages, "user_input": user_input, "response": ""}
    
    def generate_response(state: ConversationState) -> ConversationState:
        """Generate a response based on conversation history using OllamaLLM."""
        messages = state["messages"]
        user_input = state["user_input"]
        llm = OllamaLLM(model="llama3.2:1b")
        # Prepare conversation history for LLM
        chat_history = []
        for msg in messages:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))
        chat_history.append(HumanMessage(content=user_input))
        response = llm.invoke(chat_history)
        response_text = response.content if hasattr(response, 'content') else str(response)
        messages.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat()
        })
        return {"messages": messages, "user_input": user_input, "response": response_text}
    
    def display_conversation(state: ConversationState) -> ConversationState:
        """Display the conversation history."""
        print("Conversation History:")
        for msg in state["messages"]:
            print(f"{msg['role'].upper()}: {msg['content']}")
        print()
        return state
    
    # Build the conversation graph
    workflow = StateGraph(ConversationState)
    workflow.add_node("process_input", process_user_input)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("display", display_conversation)
    
    workflow.set_entry_point("process_input")
    workflow.add_edge("process_input", "generate_response")
    workflow.add_edge("generate_response", "display")
    workflow.add_edge("display", END)
    
    app = workflow.compile()
    
    # Simulate a conversation
    conversation_state = {"messages": [], "user_input": "", "response": ""}
    
    # First message
    result1 = app.invoke({
        **conversation_state,
        "user_input": "Hello, how are you?"
    })
    
    # Second message (continuing the conversation)
    result2 = app.invoke({
        **result1,
        "user_input": "What's the weather like?"
    })
    
    print("Final conversation state:")
    for msg in result2["messages"]:
        print(f"{msg['role'].upper()}: {msg['content']}\n")

# ============================================================================
# 2. STRUCTURED MEMORY - Typed State
# ============================================================================

def structured_memory_example():
    """
    Structured memory example using typed state for better organization.
    This shows how to use TypedDict for structured data management.
    """
    print("=== 2. STRUCTURED MEMORY - Typed State ===")
    
    class UserProfile(TypedDict):
        name: str
        preferences: List[str]
        last_interaction: str
    
    class StructuredState(TypedDict):
        user_profile: UserProfile
        conversation_history: List[Dict[str, str]]
        session_data: Dict[str, Any]
    
    def update_user_profile(state: StructuredState) -> StructuredState:
        """Update user profile based on conversation."""
        profile = state.get("user_profile", {
            "name": "Unknown",
            "preferences": [],
            "last_interaction": ""
        })
        
        # Extract information from conversation
        messages = state.get("conversation_history", [])
        if messages:
            last_message = messages[-1]["content"]
            
            # Simple extraction (in real app, use NLP)
            if "my name is" in last_message.lower():
                name = last_message.split("my name is")[-1].strip()
                profile["name"] = name
            
            if "like" in last_message.lower():
                preference = last_message.split("like")[-1].strip()
                if preference not in profile["preferences"]:
                    profile["preferences"].append(preference)
            
            profile["last_interaction"] = datetime.now().isoformat()
        
        return {
            **state,
            "user_profile": profile
        }
    
    def personalize_response(state: StructuredState) -> StructuredState:
        """Generate personalized response based on user profile using OllamaLLM."""
        profile = state["user_profile"]
        messages = state["conversation_history"]
        if not messages:
            return state
        last_message = messages[-1]["content"]
        llm = OllamaLLM(model="llama3.2:1b")
        # Build prompt with personalization
        prompt = f"You are a helpful assistant."
        if profile["name"] != "Unknown":
            prompt += f" The user's name is {profile['name']}."
        if profile["preferences"]:
            prompt += f" The user likes {', '.join(profile['preferences'])}."
        prompt += f"\nUser message: {last_message}\nRespond accordingly."
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        messages.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat()
        })
        return {
            **state,
            "conversation_history": messages,
            "session_data": {
                **state.get("session_data", {}),
                "last_response": response_text
            }
        }
    
    def display_structured_state(state: StructuredState) -> StructuredState:
        """Display the structured state information."""
        print("User Profile:")
        print(f"  Name: {state['user_profile']['name']}")
        print(f"  Preferences: {state['user_profile']['preferences']}")
        print(f"  Last Interaction: {state['user_profile']['last_interaction']}")
        
        print("\nSession Data:")
        for key, value in state.get("session_data", {}).items():
            print(f"  {key}: {value}")
        print()
        
        return state
    
    # Build the structured memory graph
    workflow = StateGraph(StructuredState)
    workflow.add_node("update_profile", update_user_profile)
    workflow.add_node("personalize", personalize_response)
    workflow.add_node("display", display_structured_state)
    
    workflow.set_entry_point("update_profile")
    workflow.add_edge("update_profile", "personalize")
    workflow.add_edge("personalize", "display")
    workflow.add_edge("display", END)
    
    app = workflow.compile()
    
    # Simulate structured conversation
    initial_state = {
        "user_profile": {"name": "Unknown", "preferences": [], "last_interaction": ""},
        "conversation_history": [{"role": "user", "content": "My name is Alice and I like pizza", "timestamp": ""}],
        "session_data": {}
    }
    
    result = app.invoke(initial_state)
    print("Final structured state created!\n")

# ============================================================================
# 3. PERSISTENT MEMORY - Database Storage
# ============================================================================

def persistent_memory_example():
    """
    Persistent memory example showing how to save and load state from storage.
    This demonstrates checkpointing and state persistence.
    """
    print("=== 3. PERSISTENT MEMORY - Database Storage ===")
    
    class PersistentState(TypedDict):
        user_id: str
        conversation_data: List[Dict[str, str]]
        metadata: Dict[str, Any]
    
    def save_conversation(state: PersistentState) -> PersistentState:
        """Save conversation data to persistent storage."""
        # In a real implementation, this would save to a database
        # For this example, we'll simulate with file storage
        
        user_id = state["user_id"]
        filename = f"conversation_{user_id}.json"
        
        # Save to file (simulating database)
        with open(filename, 'w') as f:
            json.dump({
                "user_id": user_id,
                "conversation_data": state["conversation_data"],
                "metadata": state["metadata"],
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"Saved conversation data to {filename}")
        return state
    
    def load_conversation(state: PersistentState) -> PersistentState:
        """Load conversation data from persistent storage."""
        user_id = state["user_id"]
        filename = f"conversation_{user_id}.json"
        
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                saved_data = json.load(f)
            
            # Merge with current state
            return {
                "user_id": user_id,
                "conversation_data": saved_data.get("conversation_data", []),
                "metadata": {**state.get("metadata", {}), **saved_data.get("metadata", {})}
            }
        else:
            print(f"No saved conversation found for user {user_id}")
            return state
    
    def add_message(state: PersistentState) -> PersistentState:
        """Add a new message to the conversation using OllamaLLM."""
        conversation_data = state.get("conversation_data", [])
        llm = OllamaLLM(model="llama3.2:1b")
        user_message = f"Message {len(conversation_data) + 1}"
        # Use LLM to generate a response
        response = llm.invoke(user_message)
        response_text = response.content if hasattr(response, 'content') else str(response)
        new_message = {
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        }
        ai_message = {
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat()
        }
        conversation_data.append(new_message)
        conversation_data.append(ai_message)
        return {
            **state,
            "conversation_data": conversation_data,
            "metadata": {
                **state.get("metadata", {}),
                "message_count": len(conversation_data),
                "last_message_time": ai_message["timestamp"]
            }
        }
    
    # Build the persistent memory graph
    workflow = StateGraph(PersistentState)
    workflow.add_node("load", load_conversation)
    workflow.add_node("add_message", add_message)
    workflow.add_node("save", save_conversation)
    
    workflow.set_entry_point("load")
    workflow.add_edge("load", "add_message")
    workflow.add_edge("add_message", "save")
    workflow.add_edge("save", END)
    
    app = workflow.compile()
    
    # Simulate persistent conversation
    user_state = {
        "user_id": "user123",
        "conversation_data": [],
        "metadata": {"session_start": datetime.now().isoformat()}
    }
    
    # First run
    result1 = app.invoke(user_state)
    
    # Second run (simulating a new session)
    result2 = app.invoke(user_state)
    
    print("Persistent memory demonstration completed!\n")

# ============================================================================
# 4. VECTOR MEMORY - Semantic Search
# ============================================================================

def vector_memory_example():
    """
    Vector memory example using embeddings for semantic search.
    This demonstrates how to store and retrieve information based on meaning.
    """
    print("=== 4. VECTOR MEMORY - Semantic Search ===")
    
    class VectorState(TypedDict):
        query: str
        knowledge_base: List[Dict[str, Any]]
        retrieved_context: List[str]
        response: str
    
    def initialize_knowledge_base(state: VectorState) -> VectorState:
        """Initialize a knowledge base with sample documents."""
        # Sample knowledge base (in real app, this would be loaded from documents)
        knowledge_base = [
            {"content": "Python is a programming language", "metadata": {"topic": "programming"}},
            {"content": "Machine learning uses algorithms to learn patterns", "metadata": {"topic": "AI"}},
            {"content": "Data structures organize data efficiently", "metadata": {"topic": "programming"}},
            {"content": "Neural networks are inspired by brain cells", "metadata": {"topic": "AI"}},
            {"content": "Databases store and retrieve information", "metadata": {"topic": "data"}}
        ]
        
        return {**state, "knowledge_base": knowledge_base}
    
    def semantic_search(state: VectorState) -> VectorState:
        """Perform semantic search on the knowledge base."""
        query = state["query"]
        knowledge_base = state["knowledge_base"]
        
        # Simple keyword-based search (in real app, use embeddings)
        query_lower = query.lower()
        retrieved_context = []
        
        for item in knowledge_base:
            content = item["content"].lower()
            if any(word in content for word in query_lower.split()):
                retrieved_context.append(item["content"])
        
        return {
            **state,
            "retrieved_context": retrieved_context
        }
    
    def generate_contextual_response(state: VectorState) -> VectorState:
        """Generate response based on retrieved context using OllamaLLM."""
        query = state["query"]
        context = state["retrieved_context"]
        llm = OllamaLLM(model="llama3.2:1b")
        if context:
            prompt = f"Context: {' '.join(context)}\nUser question: {query}\nAnswer:"
        else:
            prompt = f"User question: {query}\nAnswer:"
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        return {**state, "response": response_text}
    
    def display_vector_results(state: VectorState) -> VectorState:
        """Display the vector memory search results."""
        print(f"Query: {state['query']}")
        print(f"Retrieved Context: {state['retrieved_context']}")
        print(f"Response: {state['response']}")
        print()
        return state
    
    # Build the vector memory graph
    workflow = StateGraph(VectorState)
    workflow.add_node("init_kb", initialize_knowledge_base)
    workflow.add_node("search", semantic_search)
    workflow.add_node("generate", generate_contextual_response)
    workflow.add_node("display", display_vector_results)
    
    workflow.set_entry_point("init_kb")
    workflow.add_edge("init_kb", "search")
    workflow.add_edge("search", "generate")
    workflow.add_edge("generate", "display")
    workflow.add_edge("display", END)
    
    app = workflow.compile()
    
    # Test vector memory with different queries
    queries = [
        "What is Python?",
        "Tell me about machine learning",
        "How do databases work?"
    ]
    
    for query in queries:
        state = {"query": query, "knowledge_base": [], "retrieved_context": [], "response": ""}
        app.invoke(state)
    
    print("Vector memory demonstration completed!\n")

# ============================================================================
# 5. MULTI-AGENT MEMORY - Shared State
# ============================================================================

def multi_agent_memory_example():
    """
    Multi-agent memory example showing shared state between agents.
    This demonstrates how multiple agents can share and update memory.
    """
    print("=== 5. MULTI-AGENT MEMORY - Shared State ===")
    
    class SharedState(TypedDict):
        shared_knowledge: Dict[str, Any]
        agent_contributions: Dict[str, List[str]]
        current_task: str
        task_history: List[Dict[str, str]]
    
    def research_agent(state: SharedState) -> SharedState:
        """Research agent that gathers information."""
        task = state["current_task"]
        shared_knowledge = state.get("shared_knowledge", {})
        contributions = state.get("agent_contributions", {})
        
        # Simulate research findings
        research_findings = {
            "task": task,
            "sources": ["source1", "source2"],
            "key_points": [f"Point 1 about {task}", f"Point 2 about {task}"]
        }
        
        shared_knowledge[task] = research_findings
        contributions["research"] = contributions.get("research", []) + [f"Researched {task}"]
        
        return {
            **state,
            "shared_knowledge": shared_knowledge,
            "agent_contributions": contributions
        }
    
    def analysis_agent(state: SharedState) -> SharedState:
        """Analysis agent that processes gathered information."""
        task = state["current_task"]
        shared_knowledge = state.get("shared_knowledge", {})
        contributions = state.get("agent_contributions", {})
        
        if task in shared_knowledge:
            research_data = shared_knowledge[task]
            
            # Simulate analysis
            analysis_result = {
                "insights": [f"Insight 1 from {task}", f"Insight 2 from {task}"],
                "recommendations": [f"Recommendation 1 for {task}", f"Recommendation 2 for {task}"]
            }
            
            shared_knowledge[f"{task}_analysis"] = analysis_result
            contributions["analysis"] = contributions.get("analysis", []) + [f"Analyzed {task}"]
        
        return {
            **state,
            "shared_knowledge": shared_knowledge,
            "agent_contributions": contributions
        }
    
    def synthesis_agent(state: SharedState) -> SharedState:
        """Synthesis agent that combines all information using OllamaLLM."""
        task = state["current_task"]
        shared_knowledge = state.get("shared_knowledge", {})
        contributions = state.get("agent_contributions", {})
        task_history = state.get("task_history", [])
        llm = OllamaLLM(model="llama3.2:1b")
        # Create synthesis using LLM
        context = f"Task: {task}\nKnowledge: {shared_knowledge.get(task, {})}\nAnalysis: {shared_knowledge.get(f'{task}_analysis', {})}"
        prompt = f"Given the following information, provide a comprehensive summary and recommendations.\n{context}"
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        synthesis = {
            "task": task,
            "summary": response_text,
            "final_recommendations": ["Final recommendation 1", "Final recommendation 2"],
            "agents_involved": list(contributions.keys())
        }
        shared_knowledge[f"{task}_synthesis"] = synthesis
        contributions["synthesis"] = contributions.get("synthesis", []) + [f"Synthesized {task}"]
        task_history.append({
            "task": task,
            "completion_time": datetime.now().isoformat(),
            "agents_used": ", ".join(contributions.keys())
        })
        return {
            **state,
            "shared_knowledge": shared_knowledge,
            "agent_contributions": contributions,
            "task_history": task_history
        }
    
    def display_shared_state(state: SharedState) -> SharedState:
        """Display the shared state information."""
        print(f"Current Task: {state['current_task']}")
        print(f"Shared Knowledge Keys: {list(state.get('shared_knowledge', {}).keys())}")
        print(f"Agent Contributions: {state.get('agent_contributions', {})}")
        print(f"Task History: {len(state.get('task_history', []))} tasks completed")
        print()
        return state
    
    # Build the multi-agent memory graph
    workflow = StateGraph(SharedState)
    workflow.add_node("research", research_agent)
    workflow.add_node("analysis", analysis_agent)
    workflow.add_node("synthesis", synthesis_agent)
    workflow.add_node("display", display_shared_state)
    
    workflow.set_entry_point("research")
    workflow.add_edge("research", "analysis")
    workflow.add_edge("analysis", "synthesis")
    workflow.add_edge("synthesis", "display")
    workflow.add_edge("display", END)
    
    app = workflow.compile()
    
    # Simulate multi-agent collaboration
    initial_state = {
        "shared_knowledge": {},
        "agent_contributions": {},
        "current_task": "Market Research",
        "task_history": []
    }
    
    result = app.invoke(initial_state)
    print("Multi-agent memory demonstration completed!\n")

# ============================================================================
# 6. ADVANCED MEMORY - Custom Memory Classes
# ============================================================================

def advanced_memory_example():
    """
    Advanced memory example showing custom memory classes and complex state management.
    This demonstrates sophisticated memory patterns and custom implementations.
    """
    print("=== 6. ADVANCED MEMORY - Custom Memory Classes ===")
    
    class MemoryItem:
        """Custom memory item with metadata and importance scoring."""
        
        def __init__(self, content: str, memory_type: str, importance: float = 1.0):
            self.content = content
            self.memory_type = memory_type
            self.importance = importance
            self.created_at = datetime.now()
            self.access_count = 0
            self.last_accessed = None
        
        def access(self):
            """Mark this memory as accessed."""
            self.access_count += 1
            self.last_accessed = datetime.now()
        
        def to_dict(self):
            """Convert to dictionary for storage."""
            return {
                "content": self.content,
                "memory_type": self.memory_type,
                "importance": self.importance,
                "created_at": self.created_at.isoformat(),
                "access_count": self.access_count,
                "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
            }
    
    class AdvancedMemory:
        """Advanced memory system with multiple memory types and decay."""
        
        def __init__(self):
            self.short_term = []
            self.long_term = []
            self.episodic = []
            self.semantic = []
            self.max_short_term = 10
            self.max_long_term = 100
        
        def add_memory(self, content: str, memory_type: str, importance: float = 1.0):
            """Add a new memory item."""
            item = MemoryItem(content, memory_type, importance)
            
            if memory_type == "short_term":
                self.short_term.append(item)
                if len(self.short_term) > self.max_short_term:
                    # Move oldest to long term
                    oldest = self.short_term.pop(0)
                    self.long_term.append(oldest)
            elif memory_type == "long_term":
                self.long_term.append(item)
            elif memory_type == "episodic":
                self.episodic.append(item)
            elif memory_type == "semantic":
                self.semantic.append(item)
        
        def retrieve_memories(self, query: str, memory_type: str = None, limit: int = 5):
            """Retrieve memories based on query and type."""
            memories = []
            if query is None:
                return []
            if memory_type is None or memory_type == "short_term":
                memories.extend(self.short_term)
            if memory_type is None or memory_type == "long_term":
                memories.extend(self.long_term)
            if memory_type is None or memory_type == "episodic":
                memories.extend(self.episodic)
            if memory_type is None or memory_type == "semantic":
                memories.extend(self.semantic)
            # Simple keyword matching (in real app, use embeddings)
            query_lower = query.lower() if isinstance(query, str) else ""
            relevant_memories = []
            for memory in memories:
                if query_lower in memory.content.lower():
                    memory.access()
                    relevant_memories.append(memory)
            # Sort by importance and access count
            relevant_memories.sort(key=lambda x: (x.importance, x.access_count), reverse=True)
            return relevant_memories[:limit]
        
        def decay_memories(self):
            """Apply memory decay to long-term memories."""
            current_time = datetime.now()
            decayed_memories = []
            
            for memory in self.long_term:
                # Simple decay based on time and access
                time_diff = (current_time - memory.created_at).days
                decay_factor = max(0.1, 1.0 - (time_diff * 0.01) - (memory.access_count * 0.05))
                
                if decay_factor < 0.1:  # Memory is forgotten
                    decayed_memories.append(memory)
                else:
                    memory.importance *= decay_factor
            
            # Remove decayed memories
            for memory in decayed_memories:
                self.long_term.remove(memory)
    
    class AdvancedState(TypedDict):
        memory_system: AdvancedMemory
        current_context: str
        retrieved_memories: List[Dict[str, Any]]
        memory_stats: Dict[str, int]
    
    def process_context(state: AdvancedState) -> AdvancedState:
        """Process current context and add to memory using OllamaLLM."""
        context = state["current_context"]
        memory_system = state["memory_system"]
        llm = OllamaLLM(model="llama3.2:1b")
        # Add context to short-term memory
        memory_system.add_memory(context, "short_term", importance=1.0)
        # Use LLM to extract key concepts (simulate)
        prompt = f"Extract key concepts from: {context}"
        response = llm.invoke(prompt)
        key_concept = response.content if hasattr(response, 'content') else str(response)
        if key_concept and isinstance(key_concept, str):
            memory_system.add_memory(f"Concept: {key_concept}", "semantic", importance=1.5)
        memory_system.add_memory(f"Episode: {context}", "episodic", importance=1.2)
        return state
    
    def retrieve_relevant_memories(state: AdvancedState) -> AdvancedState:
        """Retrieve memories relevant to current context."""
        context = state["current_context"]
        memory_system = state["memory_system"]
        
        # Retrieve from all memory types
        relevant_memories = memory_system.retrieve_memories(context, limit=10)
        
        # Convert to dictionary format
        memory_dicts = [memory.to_dict() for memory in relevant_memories]
        
        return {
            **state,
            "retrieved_memories": memory_dicts
        }
    
    def update_memory_stats(state: AdvancedState) -> AdvancedState:
        """Update memory statistics."""
        memory_system = state["memory_system"]
        
        stats = {
            "short_term_count": len(memory_system.short_term),
            "long_term_count": len(memory_system.long_term),
            "episodic_count": len(memory_system.episodic),
            "semantic_count": len(memory_system.semantic),
            "total_memories": (len(memory_system.short_term) + 
                             len(memory_system.long_term) + 
                             len(memory_system.episodic) + 
                             len(memory_system.semantic))
        }
        
        return {**state, "memory_stats": stats}
    
    def display_advanced_memory(state: AdvancedState) -> AdvancedState:
        """Display advanced memory information."""
        print(f"Current Context: {state['current_context']}")
        print(f"Memory Stats: {state['memory_stats']}")
        print(f"Retrieved Memories: {len(state['retrieved_memories'])} items")
        
        for i, memory in enumerate(state['retrieved_memories'][:3]):
            print(f"  {i+1}. {memory['content']} (Type: {memory['memory_type']}, Importance: {memory['importance']:.2f})")
        print()
        
        return state
    
    # Build the advanced memory graph
    workflow = StateGraph(AdvancedState)
    workflow.add_node("process_context", process_context)
    workflow.add_node("retrieve_memories", retrieve_relevant_memories)
    workflow.add_node("update_stats", update_memory_stats)
    workflow.add_node("display", display_advanced_memory)
    
    workflow.set_entry_point("process_context")
    workflow.add_edge("process_context", "retrieve_memories")
    workflow.add_edge("retrieve_memories", "update_stats")
    workflow.add_edge("update_stats", "display")
    workflow.add_edge("display", END)
    
    app = workflow.compile()
    
    # Simulate advanced memory system
    memory_system = AdvancedMemory()
    
    # Add some initial memories
    memory_system.add_memory("Python is a programming language", "semantic", 2.0)
    memory_system.add_memory("User likes pizza", "episodic", 1.5)
    memory_system.add_memory("Meeting scheduled for tomorrow", "short_term", 1.0)
    
    initial_state = {
        "memory_system": memory_system,
        "current_context": "I need to learn Python programming",
        "retrieved_memories": [],
        "memory_stats": {}
    }
    
    result = app.invoke(initial_state)
    print("Advanced memory demonstration completed!\n")

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_all_memory_examples():
    """
    Run all memory examples in sequence to demonstrate the progression
    from basic to advanced memory concepts.
    """
    print("LangGraph Memory Concepts - Educational Reference")
    print("=" * 50)
    print("This demonstration shows the progression from basic to advanced memory concepts.\n")
    
    # Run all examples
    conversation_memory_example()
    structured_memory_example()
    persistent_memory_example()
    vector_memory_example()
    multi_agent_memory_example()
    advanced_memory_example()
    
    print("All memory examples completed!")
    print("\nKey Takeaways:")
    print("1. Conversation Memory: Maintaining chat history and context")
    print("2. Structured Memory: Using TypedDict for organized data")
    print("3. Persistent Memory: Saving and loading state from storage")
    print("4. Vector Memory: Semantic search using embeddings")
    print("5. Multi-Agent Memory: Shared state between multiple agents")
    print("6. Advanced Memory: Custom memory classes with decay and importance")

if __name__ == "__main__":
    # Run all examples
    run_all_memory_examples() 