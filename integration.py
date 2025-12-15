from perception.perception import PerceptionModule
from memory.working_memory import WorkingMemory
from memory.long_term_memory import LongTermMemory

class IntegratedSystem:
    def __init__(self):
        self.perception = PerceptionModule()
        self.working_memory = WorkingMemory()
        self.long_term_memory = LongTermMemory()

    def process_input(self, text=None, audio_duration=5):
        if text:
            nlu_output = self.perception.process_text(text)
        else:
            text = self.perception.process_audio(audio_duration)
            nlu_output = self.perception.process_text(text)

        # Store in working memory
        self.working_memory.store(nlu_output)

        # Optionally, transfer important info to LTM
        # For simplicity, store all for now
        self.long_term_memory.store(nlu_output)

        return nlu_output

    def get_context(self, query):
        wm_results = self.working_memory.retrieve(query)
        ltm_results = self.long_term_memory.retrieve(query)
        return {"working_memory": wm_results, "long_term_memory": ltm_results}

if __name__ == "__main__":
    system = IntegratedSystem()
    # Example with text
    output = system.process_input(text="I am feeling happy today.")
    print("NLU Output:", output)
    context = system.get_context("happy")
    print("Context:", context)
