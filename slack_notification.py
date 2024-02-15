from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from typing import Optional
import tyro

def send_slack_notification(token, channel, message):
    client = WebClient(token=token)
    try:
        response = client.chat_postMessage(channel=channel, text=message)
        assert response["message"]["text"] == message
    except SlackApiError as e:
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        print(f"Got an error: {e.response['error']}")

def main(
  token: Optional[str],
  channel: Optional[str],
  message: Optional[str]="Your script has finished running."
):
  send_slack_notification(token, channel, message)

# Usage

if __name__ == "__main__":
    tyro.cli(main)
