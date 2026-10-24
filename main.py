"""Run this file to prepare an isolated environment and start the GRPO benchmark."""

from pathlib import Path
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from portable.launcher import main

    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Completed checkpoints and logs remain in runs/ and logs/.", flush=True)
        raise SystemExit(130)
    except Exception as error:
        print(f"\nReasoningForge: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        if sys.platform == "win32" and len(sys.argv) == 1 and sys.stdin.isatty():
            input("Press Enter to close this window...")
        raise SystemExit(1)
