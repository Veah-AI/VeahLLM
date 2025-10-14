#!/usr/bin/env python3

"""VEAH LLM - Main Application"""

import sys
import argparse
from veah import VeahLLMForSolana

def main():
    parser = argparse.ArgumentParser(description='VEAH LLM - Solana-Native Language Model')
    parser.add_argument('--model', default='veah-7b', help='Model to load')
    parser.add_argument('--prompt', type=str, help='Prompt to generate from')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--server', action='store_true', help='Start API server')

    args = parser.parse_args()

    if args.server:
        from api import app
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
        return

    # Load model
    print(f"Loading VEAH LLM ({args.model})...")
    model = VeahLLMForSolana.from_pretrained(args.model)
    print("Model loaded successfully!")

    if args.interactive:
        print("\n🧠 VEAH LLM - Interactive Mode")
        print("Type 'quit' to exit\n")

        while True:
            prompt = input("You: ")
            if prompt.lower() in ['quit', 'exit']:
                break

            response = model.generate(prompt)
            print(f"\nVEAH: {response}\n")

    elif args.prompt:
        response = model.generate(args.prompt)
        print(response)

    else:
        print("Use --interactive for chat mode or --prompt 'your question' for single query")

if __name__ == "__main__":
    main()