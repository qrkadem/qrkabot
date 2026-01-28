#!/home/qrkadem/work/code/qrkabot/.venv/bin/python3
"""
Manual Override Tool for qrkabot

Usage:
    python qrkaoverride.py "Your message here"   - Set a manual override message
    python qrkaoverride.py --status              - Check if queue is empty
    python qrkaoverride.py --clear               - Force clear the queue
"""

import sys
import os

# Import the queue functions from qrkabot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qrkabot import set_manual_override, is_queue_empty, clear_manual_queue, MANUAL_QUEUE_FILE

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python qrkaoverride.py "Your message"  - Set override message')
        print("  python qrkaoverride.py --status        - Check queue status")
        print("  python qrkaoverride.py --clear         - Clear the queue")
        sys.exit(1)
    
    arg = sys.argv[1]
    
    if arg == "--status":
        if is_queue_empty():
            print("✓ Queue is empty. You can add a message.")
        else:
            with open(MANUAL_QUEUE_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            print(f"✗ Queue is occupied with: \"{content}\"")
            print("  Wait for it to be used or run --clear")
    
    elif arg == "--clear":
        clear_manual_queue()
        print("✓ Queue cleared.")
    
    else:
        # Treat everything as the message (join args if multiple)
        message = ' '.join(sys.argv[1:])
        
        if set_manual_override(message):
            print(f"✓ Manual override set: \"{message}\"")
            print("  The next ping will use this message.")
        else:
            print("✗ Queue is occupied! Cannot set new message.")
            print("  Use --status to see current message or --clear to force clear.")
            sys.exit(1)

if __name__ == "__main__":
    main()
