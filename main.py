"""
Main Entry Point - Initialize and run the Agentic RAG Agent
"""
from src.agent import create_agent


def main():
    """Main function to run the agent"""
    # Create agent (initializes tools and graph)
    print("[INFO] Creating agent...")
    agent = create_agent()
    print("[INFO] Agent ready!")
    
    # Interactive loop
    print("\n" + "="*60)
    print("[AGENT] Agentic RAG - Multi-Source Tool Retriever")
    print("="*60)
    print(f"Loaded {agent.get_tool_count()} tools")
    print("Ask questions about LangGraph, Agent Quality, or Abhiram")
    print("Type 'exit' or 'quit' to stop\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Get response from agent with details
            print("\n🤔 Thinking...\n")
            response, details = agent.query_with_details(user_input)
            
            # Display execution details
            if details["tools_used"]:
                print(f"🔧 Tools Used: {', '.join(details['tools_used'])}")
                print(f"📊 Total Steps: {details['total_messages']}\n")
            
            print(f"Assistant: {response}\n")
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


if __name__ == "__main__":
    main()
