from core.ethics_personalization import EthicalPersonalizedAgent

class AGI119Agent:
    def __init__(self):
        self.agent = EthicalPersonalizedAgent()

    def process_input(self, text, emotion):
        return self.agent.generate_response(text, emotion)
