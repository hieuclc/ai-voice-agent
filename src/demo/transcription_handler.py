from loguru import logger
from typing import Optional, List
from datetime import datetime
from pipecat.frames.frames import TranscriptionMessage, TranscriptionUpdateFrame
from pipecat.processors.transcript_processor import TranscriptProcessor
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError


class TranscriptHandler:
    """Handles real-time transcript processing with MongoDB persistence.

    Maintains conversation messages and saves them to MongoDB with session support.
    Allows loading previous conversations using session_id.

    Attributes:
        messages: List of all processed transcript messages in current session
        output_file: Optional path to file where transcript is saved
        session_id: Unique identifier for the conversation session
        mongo_client: MongoDB async client
        db: MongoDB database instance
        collection: MongoDB collection for transcripts
    """

    def __init__(
        self,
        session_id: str,
        mongo_uri: str = "mongodb://admin:admin123@localhost:27017",
        database_name: str = "voice_agent",
        collection_name: str = "transcripts",
        output_file: Optional[str] = None,
    ):
        """Initialize handler with MongoDB connection and session ID.

        Args:
            session_id: Unique identifier for this conversation session
            mongo_uri: MongoDB connection URI
            database_name: Name of the database to use
            collection_name: Name of the collection to store transcripts
            output_file: Optional path to output file for backup
        """
        self.messages: List[TranscriptionMessage] = []
        self.output_file: Optional[str] = output_file
        self.session_id: str = session_id

        # Setup MongoDB connection
        self.mongo_client = AsyncIOMotorClient(mongo_uri)
        self.db = self.mongo_client[database_name]
        self.collection = self.db[collection_name]

        logger.info(
            f"TranscriptHandler initialized for session '{session_id}' "
            f"with MongoDB at {mongo_uri}/{database_name}/{collection_name}"
        )
    def get_messages(self):
        return self.messages
    async def load_session(self) -> bool:
        """Load existing session from MongoDB if it exists.

        Returns:
            True if session was loaded, False if new session
        """
        try:
            session_doc = await self.collection.find_one({"session_id": self.session_id})

            if session_doc and "messages" in session_doc:
                # Reconstruct TranscriptionMessage objects from stored data
                self.messages = []
                for msg_data in session_doc["messages"]:
                    msg = TranscriptionMessage(
                        role=msg_data["role"],
                        content=msg_data["content"],
                        timestamp=msg_data.get("timestamp", ""),
                    )
                    self.messages.append(msg)

                logger.info(
                    f"Loaded {len(self.messages)} messages for session '{self.session_id}'"
                )
                return True
            else:
                logger.info(f"No existing session found for '{self.session_id}', starting new")
                return False

        except PyMongoError as e:
            logger.error(f"Error loading session from MongoDB: {e}")
            return False

    async def save_messages(self):
        """Save current messages to MongoDB.

        Creates or updates the session document with all messages.
        """
        try:
            # Convert messages to serializable format
            messages_data = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp if hasattr(msg, "timestamp") else "",
                }
                for msg in self.messages
            ]

            session_doc = {
                "session_id": self.session_id,
                "messages": messages_data,
                "updated_at": datetime.utcnow(),
            }

            # Upsert: update if exists, insert if not
            result = await self.collection.update_one(
                {"session_id": self.session_id},
                {"$set": session_doc},
                upsert=True,
            )

            logger.debug(
                f"Saved {len(messages_data)} messages for session '{self.session_id}' "
                f"(matched: {result.matched_count}, modified: {result.modified_count})"
            )

            # Optional: also save to file if specified
            if self.output_file:
                await self._save_to_file()

        except PyMongoError as e:
            logger.error(f"Error saving messages to MongoDB: {e}")

    async def _save_to_file(self):
        """Save messages to output file as backup."""
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                for msg in self.messages:
                    timestamp = getattr(msg, "timestamp", "")
                    f.write(f"[{timestamp}] {msg.role}: {msg.content}\n")
            logger.debug(f"Backup saved to file: {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving to file: {e}")

    async def on_transcript_update(
        self, processor: TranscriptProcessor, frame: TranscriptionUpdateFrame
    ):
        """Handle new transcript messages and save to MongoDB.

        Args:
            processor: The TranscriptProcessor that emitted the update
            frame: TranscriptionUpdateFrame containing new messages
        """
        logger.debug(
            f"Received transcript update with {len(frame.messages)} new messages"
        )

        for msg in frame.messages:
            self.messages.append(msg)

        # Auto-save after each update
        await self.save_messages()

    async def get_context(self, max_messages: Optional[int] = None) -> List[dict]:
        """Get conversation context for LLM.

        Args:
            max_messages: Maximum number of recent messages to return. None for all.

        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        messages_to_return = self.messages[-max_messages:] if max_messages else self.messages

        return [{"role": msg.role, "content": msg.content} for msg in messages_to_return]

    async def clear_session(self):
        """Clear current session from memory and database."""
        try:
            await self.collection.delete_one({"session_id": self.session_id})
            self.messages.clear()
            logger.info(f"Cleared session '{self.session_id}'")
        except PyMongoError as e:
            logger.error(f"Error clearing session: {e}")

    async def close(self):
        """Close MongoDB connection gracefully."""
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("MongoDB connection closed")