#!/usr/bin/env python

from langserve import RemoteRunnable
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Joke Generator Client')
    parser.add_argument('--topic', type=str, default='programming',
                       help='Topic to generate a joke about')
    args = parser.parse_args()
    
    # Connect to the remote server
    remote_chain = RemoteRunnable("http://localhost:8000/joke-generator/")
    
    # Display what we're doing
    print(f"Generating a joke about: {args.topic}")
    
    # Invoke the remote chain
    try:
        response = remote_chain.invoke({"topic": args.topic})
        print("\nJoke Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
    except Exception as e:
        print(f"Error connecting to server: {e}")
        print("Make sure server.py is running on http://localhost:8000")

if __name__ == "__main__":
    main()