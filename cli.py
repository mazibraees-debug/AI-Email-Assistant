"""Quick terminal demo:  python cli.py"""
import json
from pathlib import Path

from assistant import EmailAssistant, from_text

bot = EmailAssistant()
emails = [from_text(**d) for d in json.loads((Path(__file__).parent / "data/sample_emails.json").read_text())]

for email, a in bot.process_inbox(emails):
    print(f"\n[{a.priority}/5] {a.category}: {email.subject}")
    print(f"  Summary: {a.summary}")
    for item in a.action_items:
        print(f"  - {item}")

pick = input("\nGenerate a reply for the top email? (y/n) ")
if pick.lower() == "y":
    top, ana = bot.process_inbox(emails)[0]
    print("\n" + bot.draft_reply(top, ana))
    print("\n(Review it, then send manually - the CLI never sends.)")
