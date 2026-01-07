import random
import re
import sys
from twisted.internet import protocol, reactor
from twisted.words.protocols import irc
# hey look! it's qrkabot!
from qrkabot import generate_response

class qrkabot(irc.IRCClient):
    nickname = "qrkabot"

    def lineReceived(self, line):
        print(line.decode('utf-8', errors='ignore'))
        super().lineReceived(line)

    def sendwithlag(self, channel, text):
        lines = text.split("\n")
        for i, line in enumerate(lines):
            reactor.callLater(random.uniform(0.3, 0.6), self.msg, channel, line)

    def connectionMade(self):
        irc.IRCClient.connectionMade(self)
        print("Connected ✓")

    def signedOn(self):
        self.join(self.factory.channel)

    def joined(self, channel):
        print(f"Joined {channel}")

    def privmsg(self, user, channel, msg):
        u = user.split('!')[0]
        is_pm = (channel == self.nickname)

        # only require mentions in channels
        if not is_pm and self.nickname.lower() not in msg.lower():
            return

        prompt = msg

        if not is_pm:
            # existing cleanup, channel only
            prompt = re.sub(r"^\[[^\]]+\]\s*", "", prompt)
            prompt = re.sub(r"<[^>]+>\s*", "", prompt, count=1)
            prompt = re.sub(
                re.escape(self.nickname),
                "",
                prompt,
                count=1,
                flags=re.IGNORECASE
            )
            prompt = prompt.strip()

        reply_target = u if is_pm else channel

        try:
            response = generate_response(prompt, user=u)
            self.sendwithlag(channel, response)
        except Exception as e:
            print(f"Error generating response: {e}")
            self.msg(
                reply_target,
                random.choice([
                    "Sorry, I couldn't generate a response.",
                    "AAAA! Tell qrkadem to fix his code ",
                    "seg fault",
                    "QRKADEM FIX YOUR CODE",
                    "qrkadem: you idiot",
                ])
            )
class qrkabotFactory(protocol.ClientFactory):
    protocol = qrkabot

    def __init__(self, channel):
        self.channel = channel

    def clientConnectionLost(self, connector, reason):
        print("Connection lost")
        reactor.stop()

    def clientConnectionFailed(self, connector, reason):
        print("Connection failed")
        reactor.stop()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qrkaclient.py <channel>")
        sys.exit(1)

    server = "irc.choopa.net"
    port = 6667
    channel = sys.argv[1]

    factory = qrkabotFactory(channel)
    reactor.connectTCP(server, port, factory)
    reactor.run()