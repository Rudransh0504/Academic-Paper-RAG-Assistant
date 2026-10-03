from src.rag.generate import generate, MODEL

print("Model:", MODEL)
print(generate("You are helpful.", "Say hello in one short sentence."))