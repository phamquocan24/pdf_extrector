#!/usr/bin/env python3
"""
Test LLM processing and fallback mechanisms
"""

print("🧪 Testing LLM Processing & Fallback")
print("=" * 50)

try:
    import utils
    print("✅ Utils imported successfully")
    
    # Test LLM processing function
    if hasattr(utils, 'process_text_with_llm'):
        
        # Test cases
        test_cases = [
            "2002\\n \\n2001\\n \\n2000",
            "7,500",
            "benefit cost\\n",
            "",
            "   multiple   spaces   text   "
        ]
        
        print(f"\n🧠 Testing LLM Text Processing:")
        print(f"Current LLM Status:")
        print(f"  - Groq client: {'✅' if utils.groq_client else '❌'}")
        print(f"  - OpenAI client: {'✅' if utils.openai_client else '❌'}")
        
        for i, test_text in enumerate(test_cases, 1):
            if not test_text:
                continue
                
            print(f"\n{i}. Testing: '{test_text}'")
            try:
                result = utils.process_text_with_llm(test_text, "test")
                print(f"   Result: '{result}'")
                
                if result != test_text:
                    print(f"   ✅ Text was processed/cleaned")
                else:
                    print(f"   ℹ️  Text unchanged (no cleaning needed)")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Test empty text
        print(f"\n6. Testing empty text:")
        empty_result = utils.process_text_with_llm("", "test")
        print(f"   Result: '{empty_result}' (should be empty)")
        
    else:
        print("❌ process_text_with_llm function not found")
    
    print(f"\n🎯 Extraction Pipeline Summary:")
    if hasattr(utils, 'get_extraction_capabilities'):
        capabilities, methods = utils.get_extraction_capabilities()
        for i, method in enumerate(methods, 1):
            status = "✅" if ("disabled" not in method.lower()) else "⚠️"
            print(f"  {i}. {status} {method}")
            
        print(f"\n📊 Capabilities Details:")
        for key, value in capabilities.items():
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")
    
    print(f"\n🎉 LLM test completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "=" * 50)
print(f"💡 Expected behavior:")
print(f"  - If Groq API key invalid: Falls back to basic cleaning")
print(f"  - If no LLM available: Uses basic text cleaning")  
print(f"  - System should never crash due to LLM errors")
print(f"  - Text processing should always return some result")
