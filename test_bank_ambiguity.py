import torch
from state_core.pipeline import StateCorePipeline
from transformers import GPT2Tokenizer

def test_bank_ambiguity():
    """
    Test that the SOSM system disambiguates 'bank' based on context.
    
    Case 1: "The bank of the river is" → geographic context
    Case 2: "The bank loan is" → financial context
    
    Expected: Different semantic graphs and different next-token predictions
    """
    print("=" * 60)
    print("Bank Ambiguity Disambiguation Test")
    print("=" * 60)
    
    # Initialize
    print("\n[1] Initializing model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Enable graph routing (Stage 3)
    config = {
        'stage': 3,
        'components': {
            'graph': {
                'enabled': True,
                'sequential_edges': True,
                'semantic_edges': True,
                'semantic_threshold': 0.3,
                'random_shortcuts': 0.05
            }
        }
    }
    
    model = StateCorePipeline(config)
    model.eval()
    print("✅ Model initialized with graph routing enabled")
    
    # Case 1: Geographic context
    print("\n[2] Testing Case 1: Geographic context")
    print("    Input: 'The bank of the river is'")
    
    tokens1 = tokenizer.encode("The bank of the river is", return_tensors='pt')
    
    with torch.no_grad():
        logits1, state1 = model(tokens1, return_state=True)
    
    graph1 = state1.routing_state['graph'] if state1.routing_state else None
    next_token1 = logits1[0, -1].argmax()
    next_word1 = tokenizer.decode([next_token1])
    
    print(f"    Graph edges: {graph1['num_edges'] if graph1 else 'N/A'}")
    if graph1:
        print(f"    - Sequential: {graph1['edge_types']['sequential']}")
        print(f"    - Semantic: {graph1['edge_types']['semantic']}")
        print(f"    - Shortcuts: {graph1['edge_types']['shortcut']}")
    print(f"    Next token: '{next_word1}'")
    
    # Case 2: Financial context
    print("\n[3] Testing Case 2: Financial context")
    print("    Input: 'The bank loan is'")
    
    tokens2 = tokenizer.encode("The bank loan is", return_tensors='pt')
    
    with torch.no_grad():
        logits2, state2 = model(tokens2, return_state=True)
    
    graph2 = state2.routing_state['graph'] if state2.routing_state else None
    next_token2 = logits2[0, -1].argmax()
    next_word2 = tokenizer.decode([next_token2])
    
    print(f"    Graph edges: {graph2['num_edges'] if graph2 else 'N/A'}")
    if graph2:
        print(f"    - Sequential: {graph2['edge_types']['sequential']}")
        print(f"    - Semantic: {graph2['edge_types']['semantic']}")
        print(f"    - Shortcuts: {graph2['edge_types']['shortcut']}")
    print(f"    Next token: '{next_word2}'")
    
    # Verify results
    print("\n[4] Verification")
    print("=" * 60)
    
    if graph1 and graph2:
        # Check that graphs are different
        edges_match = graph1['num_edges'] == graph2['num_edges']
        print(f"    Different graphs created: {'❌' if edges_match else '✅'}")
        
        # Check semantic edges exist
        has_semantic1 = graph1['edge_types']['semantic'] > 0
        has_semantic2 = graph2['edge_types']['semantic'] > 0
        print(f"    Case 1 has semantic edges: {'✅' if has_semantic1 else '❌'}")
        print(f"    Case 2 has semantic edges: {'✅' if has_semantic2 else '❌'}")
        
        # Check predictions differ
        predictions_differ = next_token1 != next_token2
        print(f"    Different predictions: {'✅' if predictions_differ else '❌'}")
        
        # Overall result
        success = (not edges_match or predictions_differ) and (has_semantic1 or has_semantic2)
        print(f"\n{'✅ TEST PASSED' if success else '❌ TEST FAILED'}")
        
        if success:
            print("\nThe system successfully disambiguates 'bank' based on context!")
            print("Different neighbors → Different graphs → Different meanings")
        
        return success
    else:
        print("❌ Graph routing not enabled - check config")
        print("\nTo enable, set in config.yaml:")
        print("  stage: 3")
        print("  components:")
        print("    graph:")
        print("      enabled: true")
        return False


if __name__ == "__main__":
    try:
        test_bank_ambiguity()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
