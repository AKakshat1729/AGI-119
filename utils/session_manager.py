"""
Session Manager - Handles conversation session lifecycle
Automatically saves sessions to long-term memory when new sessions start
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import json

class SessionManager:
    def __init__(self, memory_store, llm_client):
        """
        Initialize session manager
        Args:
            memory_store: ServerMemoryStore instance
            llm_client: LLM client for generating summaries
        """
        self.memory_store = memory_store
        self.llm_client = llm_client
        self.active_sessions = {}  # {user_id: {conversation_id, start_time, message_count}}
    
    def start_new_session(self, user_id: str, conversation_id: str = None) -> str:
        """
        Start a new conversation session
        If user has an active session, archive it first
        
        Args:
            user_id: User identifier
            conversation_id: Optional conversation ID (generates new if None)
            
        Returns:
            conversation_id for the new session
        """
        # Check if user has an active session
        if user_id in self.active_sessions:
            old_session = self.active_sessions[user_id]
            # Archive the old session if it has messages
            if old_session.get('message_count', 0) > 0:
                self._archive_session(user_id, old_session)
        
        # Create new session
        new_conversation_id = conversation_id or str(uuid.uuid4())
        self.active_sessions[user_id] = {
            'conversation_id': new_conversation_id,
            'start_time': datetime.now().isoformat(),
            'message_count': 0,
            'first_message': None
        }
        
        print(f"[SESSION] Started new session for user {user_id}: {new_conversation_id[:8]}...")
        return new_conversation_id
    
    def track_message(self, user_id: str, conversation_id: str, message: str, is_user: bool = True):
        """
        Track a message in the current session
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            message: Message text
            is_user: True if user message, False if AI response
        """
        # Ensure session exists
        if user_id not in self.active_sessions or \
           self.active_sessions[user_id]['conversation_id'] != conversation_id:
            # Session doesn't exist or mismatch, create it
            self.active_sessions[user_id] = {
                'conversation_id': conversation_id,
                'start_time': datetime.now().isoformat(),
                'message_count': 0,
                'first_message': None
            }
        
        session = self.active_sessions[user_id]
        session['message_count'] += 1
        
        # Store first user message for summary
        if is_user and session['first_message'] is None:
            session['first_message'] = message[:100]  # First 100 chars
    
    def _archive_session(self, user_id: str, session: Dict):
        """
        Archive a completed session to long-term memory
        """
        try:
            conversation_id = session['conversation_id']
            # Get all messages
            messages = self.memory_store.get_conversation_messages(user_id, conversation_id)
            if not messages: return

            # Generate concise fact-based summary (Max 30 tokens)
            summary = self._generate_session_summary(messages)
            
            # Store ONLY the concise fact string in episodic memory
            # This ensures retrieval fetches only concrete facts
            self.memory_store.store_memory(
                user_id=user_id,
                memory_type="episodic",
                text=summary, 
                tags=["session_fact", f"conv_{conversation_id}"],
                importance=1.0
            )
            
            print(f"[SESSION] Archived facts: {summary}")
            
        except Exception as e:
            print(f"[SESSION] Error archiving: {e}")
    
    def _generate_session_summary(self, messages: List[Dict]) -> str:
        """
        Generate <30 token summary of concrete facts
        """
        try:
            # Prepare transcript for LLM
            transcript = "\n".join([f"{m.get('role', 'unknown')}: {m.get('text', '')}" for m in messages])
            
            # Ask LLM for strict fact extraction
            prompt = [
                {"role": "system", "content": "You are a data compressor. Extract ONLY concrete facts (names, dates, numbers, specific conditions) from the conversation. Output must be a SINGLE sentence under 30 words. No filler words like 'User discussed' or 'The conversation was about'. Just facts."},
                {"role": "user", "content": transcript}
            ]
            
            response = self.llm_client(prompt, max_tokens=50) # Low token limit
            
            if isinstance(response, dict) and 'response' in response:
                return response['response'].strip()
            return "Session completed."
            
        except Exception as e:
            print(f"[SESSION] Summary error: {e}")
            return "Session data stored."
    
    def end_session(self, user_id: str, conversation_id: str = None):
        """
        Explicitly end a session and archive it
        
        Args:
            user_id: User identifier
            conversation_id: Optional conversation ID to end (uses active if None)
        """
        if user_id in self.active_sessions:
            session = self.active_sessions[user_id]
            
            # If conversation_id specified, verify it matches
            if conversation_id and session['conversation_id'] != conversation_id:
                print(f"[SESSION] Warning: Conversation ID mismatch")
                return
            
            # Archive the session
            if session.get('message_count', 0) > 0:
                self._archive_session(user_id, session)
            
            # Remove from active sessions
            del self.active_sessions[user_id]
            print(f"[SESSION] Ended session for user {user_id}")
    
    def get_active_session(self, user_id: str) -> Optional[Dict]:
        """
        Get active session for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Session dictionary or None
        """
        return self.active_sessions.get(user_id)
    
    def get_session_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """
        Get archived session history for a user
        
        Args:
            user_id: User identifier
            limit: Maximum number of sessions to return
            
        Returns:
            List of session summaries
        """
        try:
            # Retrieve episodic memories tagged as session archives
            memories = self.memory_store.retrieve_memories(
                user_id=user_id,
                query="session conversation summary",
                memory_type="episodic",
                tags=["session_archive"],
                top_k=limit
            )
            
            sessions = []
            for mem in memories:
                sessions.append({
                    'summary': mem['text'],
                    'timestamp': mem.get('metadata', {}).get('timestamp'),
                    'id': mem.get('id')
                })
            
            return sessions
            
        except Exception as e:
            print(f"[SESSION] Error retrieving session history: {e}")
            return []
