from rag_engine import RAGEngine

def test():
    engine = RAGEngine()
    
    # Test Query
    query = "If I had to choose only two workouts for my chest, what would those be?"
    print(f"\n--- Testing Query: {query} ---")
    
    # 1. Test Router
    genre = engine.route_query(query)
    print(f"Detected Genre: {genre}")
    
    # 2. Test Full Pipeline
    result = engine.generate_answer(query, genre)
    
    print("\n--- ANSWER ---")
    print(result['answer'])
    
    print("\n--- METADATA ---")
    print(f"Total Cost: ${result['total_cost']:.5f}")
    print(f"Sources: {result['sources']}")

if __name__ == "__main__":
    test()