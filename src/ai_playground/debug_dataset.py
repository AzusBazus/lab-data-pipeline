from prepare_dataset import generate_examples

print("🕵️‍♀️ Starting Debug Run...")

# We manually call the generator function
generator = generate_examples()

try:
    for i, example in enumerate(generator):
        print(f"✅ Processed item {i}: {example['id']}")
        # Stop after 5 items to save time
        if i >= 5:
            break
            
except Exception as e:
    print("\n\n❌ CRASH DETECTED!")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {e}")
    
    # Check if it's a label mismatch (Most common error)
    import traceback
    traceback.print_exc()