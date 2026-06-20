import argparse
import sys
from .proxy import start_proxy

def main():
    """
    Main CLI entrypoint.
    Runs commands like:
      token-diet proxy --port 8787
    """
    parser = argparse.ArgumentParser(
        prog="token-diet",
        description="Token Diet - The lightweight context-compression layer for LLM agents"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")
    
    # Subcommand: proxy
    proxy_parser = subparsers.add_parser("proxy", help="Start the local HTTP optimization proxy server")
    proxy_parser.add_argument("--host", default="127.0.0.1", help="Host IP to bind uvicorn server (default: 127.0.0.1)")
    proxy_parser.add_argument("--port", type=int, default=8787, help="Port to listen for incoming requests (default: 8787)")
    proxy_parser.add_argument("--threshold", type=int, default=1000, help="Compression threshold in characters (default: 1000)")

    args = parser.parse_args()

    if args.command == "proxy":
        try:
            start_proxy(host=args.host, port=args.port, threshold=args.threshold)
        except KeyboardInterrupt:
            print("\nProxy server stopped by user.")
            sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
