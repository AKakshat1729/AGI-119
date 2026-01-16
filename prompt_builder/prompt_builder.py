import tiktoken
from typing import List, Dict, Any

class PromptBuilder:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.encoding = tiktoken.get_encoding("cl100k_base")  # Use cl100k_base for GPT models
        self.max_tokens = 4096  # For gpt-3.5-turbo

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        tokens = self.encoding.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = self.encoding.decode(tokens)
        return text

    def build_prompt(self, user_id: str, current_message: str, retrieved_bundle: Dict[str, Any], style_config: Dict[str, Any]) -> Dict[str, Any]:
        style = style_config.get('style', 'medium')  # short, medium, long
        therapeutic = style_config.get('therapeutic', True)

        # System prompt
        system_prompt = """
You are a compassionate AI therapist assistant. Provide empathetic, non-judgmental support.
If the user shows signs of crisis, encourage professional help.
Do not diagnose or prescribe.
"""

        if therapeutic:
            system_prompt += " Respond in a therapeutic manner."

        # Profile summary
        profile = retrieved_bundle.get('profile_summary', '')
        profile_section = f"User Profile Summary:\n{profile}\n\n" if profile else ""

        # Episodic memories
        top_memories = retrieved_bundle.get('top_memories', [])
        episodic_section = "Relevant Past Memories:\n"
        for mem in top_memories:
            episodic_section += f"- {mem['text']}\n"
        episodic_section += "\n"

        # Risk flags
        risk_flags = retrieved_bundle.get('risk_flags', [])
        risk_section = ""
        if risk_flags:
            risk_section = "Risk Flags: " + ", ".join(risk_flags) + ". Handle with care.\n\n"

        # Current message
        user_message = f"User: {current_message}\n"

        # Combine
        full_prompt = system_prompt + "\n\n" + profile_section + episodic_section + risk_section + user_message

        # Token budgeting
        token_count = self.count_tokens(full_prompt)
        if token_count > self.max_tokens - 200:  # Reserve for response
            # Truncate episodic section
            max_episodic_tokens = self.max_tokens - 200 - self.count_tokens(system_prompt + profile_section + risk_section + user_message)
            episodic_text = "\n".join([f"- {mem['text']}" for mem in top_memories])
            episodic_text = self.truncate_text(episodic_text, max_episodic_tokens)
            episodic_section = "Relevant Past Memories:\n" + episodic_text + "\n"
            full_prompt = system_prompt + "\n\n" + profile_section + episodic_section + risk_section + user_message
            token_count = self.count_tokens(full_prompt)

        # Messages for LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": profile_section + episodic_section + risk_section + user_message}
        ]

        # Debug
        debug_prompt_text = full_prompt

        return {
            "messages": messages,
            "debug_prompt_text": debug_prompt_text,
            "token_count": token_count
        }