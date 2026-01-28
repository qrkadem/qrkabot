import random
import re
import sys
from twisted.internet import protocol, reactor
from twisted.words.protocols import irc
# hey look! it's qrkabot!
from qrkabot import generate_response

class qrkabot(irc.IRCClient):
    nickname = "qrkabot"


    def sendLine(self, line):
        try:
            print(f">>> {line.decode('utf-8', errors='ignore')}")
        except Exception:
            print(f">>> {line!r}")
        super().sendLine(line)

    def lineReceived(self, line):
        print(line.decode('utf-8', errors='ignore'))
        super().lineReceived(line)

    def sendwithlag(self, channel, text):
        delay = 0.0
        for line in text.split("\n"):
            delay += random.uniform(0.6, 1.2)
            reactor.callLater(delay, self.msg, channel, line)

    def connectionMade(self):
        irc.IRCClient.connectionMade(self)
        print("Connected ✓")

    def signedOn(self):
        for channel in self.factory.channels:
            self.join(channel)

    def joined(self, channel):
        print(f"Joined {channel}")

    def privmsg(self, user, channel, msg):
        u = user.split('!')[0]
        is_pm = (channel == self.nickname)

        # Check if we should respond
        mentioned = self.nickname.lower() in msg.lower()
        
        # 1 in 300 chance to respond even without being pinged (channels only)
        random_activation = not is_pm and not mentioned and random.randint(1, 300) == 1
        
        # only require mentions in channels (unless random activation)
        if not is_pm and not mentioned and not random_activation:
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
            self.sendwithlag(reply_target, response)
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

    def __init__(self, channels):
        self.channels = channels

    def clientConnectionLost(self, connector, reason):
        print("Connection lost")
        reactor.stop()

    def clientConnectionFailed(self, connector, reason):
        print("Connection failed")
        reactor.stop()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qrkaclient.py <channel1> <channel2> ...")
        sys.exit(1)

    server = "irc.colosolutions.net"
    port = 6667
    channels = sys.argv[1:]

    factory = qrkabotFactory(channels)
    reactor.connectTCP(server, port, factory)
    reactor.run()