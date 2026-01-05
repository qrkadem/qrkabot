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

    def connectionMade(self):
        irc.IRCClient.connectionMade(self)
        print("Connected ✓")

    def signedOn(self):
        self.join(self.factory.channel)

    def joined(self, channel):
        print(f"Joined {channel}")

    def privmsg(self, user, channel, msg):
        user = user.split('!')[0]

        if channel == self.nickname:
            return  # ignore PMs for now

        if self.nickname.lower() not in msg.lower():
            return
        cleaned = re.sub(r"<[^>]+>\s*", "", msg, count=1)

        # remove first occurrence of bot nickname
        cleaned = re.sub(
            re.escape(self.nickname),
            "",
            cleaned,
            count=1,
            flags=re.IGNORECASE
        ).strip()

        prompt = cleaned

        try:
            response = generate_response(prompt)
            self.msg(channel, response)
        except Exception as e:
            print(f"Error generating response: {e}")
            self.msg(channel, "Sorry, I couldn't generate a response.")

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

    server = "irc.colosolutions.net"
    port = 6667
    channel = sys.argv[1]

    factory = qrkabotFactory(channel)
    reactor.connectTCP(server, port, factory)
    reactor.run()