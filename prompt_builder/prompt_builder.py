import json

class PromptBuilder:
    def __init__(self, model="llama3-70b-8192"):
        self.model = model

    def build_prompt(self, user_id, transcript, retrieved_memories, style_config, reasoning_data):
        """
        Converts complex JSON data into a concise text prompt to save tokens.
        """
        
        # 1. THE PERSONA (System Prompt)
        system_instruction = (
            "You are an empathetic, professional AI therapist. "
            "Your goal is to provide a response in a way which helps a person reach his desired goal , making sure user's opinion doesn't get re-enforced if the user's beliefs are diverging from the real world "
            "Keep responses concise (under 3 sentences) unless the user asks for detail. "
            "Do not start with 'I understand' every time. Be natural."
        )

        # 2. THE CONTEXT (Dynamic Construction)
        context_block = ""

        # A. Emotional State (From Perception)
        # We assume reasoning_data might contain the tone/emotion analysis
        current_emotion = "neutral"
        if reasoning_data and 'therapeutic_insight' in reasoning_data:
             # Extract emotion if hidden in insight, or pass it explicitly if you change app.py
             pass 

        # B. Relevant History (From Memory)
        # retrieved_memories is usually a dict with 'top_memories' list
        history_text = ""
        if retrieved_memories and 'top_memories' in retrieved_memories:
            memories = retrieved_memories['top_memories']
            if memories:
                history_text = "Relevant Past Context:\n"
                for mem in memories[:3]: # Limit to top 3 to save tokens
                    # If memory is a dict, get 'text', else use string
                    val = mem.get('text') if isinstance(mem, dict) else str(mem)
                    history_text += f"- {val}\n"

        # C. Deep Insight (From Reasoning)
        insight_text = ""
        if reasoning_data:
            # Check for life story facts
            life_story = reasoning_data.get('life_story', {})
            if life_story and 'potential_facts' in life_story:
                facts = life_story['potential_facts']
                if facts:
                    insight_text += f"User Facts: {', '.join(facts[:3])}.\n"
            
            # Check for therapeutic insight
            if 'therapeutic_insight' in reasoning_data:
                insight_text += f"Therapeutic Note: {reasoning_data['therapeutic_insight']}\n"

        # 3. ASSEMBLE THE FINAL PROMPT
        # We format it as a list of messages for the Chat API
        
        user_message_content = f"""
        {insight_text}
        {history_text}
        
        User's Current Input: "{transcript}"
        
        Respond to the user now, incorporating the context above naturally.
        """

        # Clean up extra whitespace
        user_message_content = "\n".join([line.strip() for line in user_message_content.split('\n') if line.strip()])

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_message_content}
        ]

        # Return the payload expected by your app.py
        return {
            "model": self.model,
            "messages": messages,
            "token_count": len(user_message_content) // 4 # Rough estimate
        }