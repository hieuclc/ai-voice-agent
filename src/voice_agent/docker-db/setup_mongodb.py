# setup_mongodb.py
"""
Script để setup MongoDB schema và indexes.
Chạy script này nếu muốn setup lại database hoặc migrate schema.
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import CollectionInvalid
from loguru import logger
from datetime import datetime


class MongoDBSetup:
    def __init__(
        self,
        mongo_uri: str = "mongodb://admin:admin123@localhost:27017",
        database_name: str = "voice_agent"
    ):
        self.mongo_uri = mongo_uri
        self.database_name = database_name
        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client[database_name]

    async def create_transcripts_collection(self):
        """Create transcripts collection with validation schema."""
        try:
            await self.db.create_collection(
                'transcripts',
                validator={
                    '$jsonSchema': {
                        'bsonType': 'object',
                        'required': ['session_id', 'messages', 'updated_at'],
                        'properties': {
                            'session_id': {
                                'bsonType': 'string',
                                'description': 'Unique session identifier - required'
                            },
                            'messages': {
                                'bsonType': 'array',
                                'description': 'Array of conversation messages - required',
                                'items': {
                                    'bsonType': 'object',
                                    'required': ['role', 'content'],
                                    'properties': {
                                        'role': {
                                            'bsonType': 'string',
                                            'enum': ['user', 'assistant', 'system'],
                                            'description': 'Message role'
                                        },
                                        'content': {
                                            'bsonType': 'string',
                                            'description': 'Message content'
                                        },
                                        'timestamp': {
                                            'bsonType': 'string',
                                            'description': 'ISO timestamp'
                                        }
                                    }
                                }
                            },
                            'updated_at': {
                                'bsonType': 'date',
                                'description': 'Last update timestamp'
                            },
                            'created_at': {
                                'bsonType': 'date',
                                'description': 'Session creation timestamp'
                            },
                            'metadata': {
                                'bsonType': 'object',
                                'description': 'Additional session metadata'
                            }
                        }
                    }
                }
            )
            logger.info("✅ Collection 'transcripts' created with validation schema")
        except CollectionInvalid:
            logger.warning("Collection 'transcripts' already exists")

    async def create_analytics_collection(self):
        """Create analytics collection."""
        try:
            await self.db.create_collection(
                'analytics',
                validator={
                    '$jsonSchema': {
                        'bsonType': 'object',
                        'required': ['session_id', 'event_type', 'timestamp'],
                        'properties': {
                            'session_id': {
                                'bsonType': 'string',
                                'description': 'Related session ID'
                            },
                            'event_type': {
                                'bsonType': 'string',
                                'enum': ['session_start', 'session_end', 'error', 'user_feedback'],
                                'description': 'Type of analytics event'
                            },
                            'timestamp': {
                                'bsonType': 'date',
                                'description': 'Event timestamp'
                            },
                            'data': {
                                'bsonType': 'object',
                                'description': 'Event-specific data'
                            }
                        }
                    }
                }
            )
            logger.info("✅ Collection 'analytics' created")
        except CollectionInvalid:
            logger.warning("Collection 'analytics' already exists")

    async def create_indexes(self):
        """Create indexes for better query performance."""
        # Transcripts indexes
        await self.db.transcripts.create_index('session_id', unique=True)
        await self.db.transcripts.create_index([('updated_at', -1)])
        await self.db.transcripts.create_index('metadata.user_id')
        await self.db.transcripts.create_index([('created_at', -1)])
        logger.info("✅ Indexes created for 'transcripts' collection")

        # Analytics indexes
        await self.db.analytics.create_index('session_id')
        await self.db.analytics.create_index('event_type')
        await self.db.analytics.create_index([('timestamp', -1)])
        logger.info("✅ Indexes created for 'analytics' collection")

    async def insert_sample_data(self):
        """Insert sample data for testing."""
        sample_transcript = {
            'session_id': 'sample_session_001',
            'messages': [
                {
                    'role': 'user',
                    'content': 'Xin chào, bạn có thể giúp tôi không?',
                    'timestamp': datetime.utcnow().isoformat()
                },
                {
                    'role': 'assistant',
                    'content': 'Chào bạn! Tôi có thể giúp gì cho bạn?',
                    'timestamp': datetime.utcnow().isoformat()
                }
            ],
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'metadata': {
                'user_id': 'user_123',
                'language': 'vi',
                'total_duration': 0
            }
        }

        result = await self.db.transcripts.update_one(
            {'session_id': 'sample_session_001'},
            {'$set': sample_transcript},
            upsert=True
        )
        logger.info(f"✅ Sample data inserted (modified: {result.modified_count})")

    async def verify_setup(self):
        """Verify that collections and indexes are properly set up."""
        collections = await self.db.list_collection_names()
        logger.info(f"📋 Collections: {collections}")

        for collection_name in ['transcripts', 'analytics']:
            if collection_name in collections:
                indexes = await self.db[collection_name].index_information()
                logger.info(f"📊 Indexes for '{collection_name}': {list(indexes.keys())}")

        # Test query
        count = await self.db.transcripts.count_documents({})
        logger.info(f"📈 Total transcripts: {count}")

    async def setup_all(self):
        """Run complete setup process."""
        logger.info("🚀 Starting MongoDB setup...")

        await self.create_transcripts_collection()
        await self.create_analytics_collection()
        await self.create_indexes()
        await self.insert_sample_data()
        await self.verify_setup()

        logger.info("✨ MongoDB setup completed successfully!")

    async def drop_all(self):
        """Drop all collections (use with caution!)."""
        confirm = input("⚠️  This will delete all data. Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            await self.db.transcripts.drop()
            await self.db.analytics.drop()
            logger.warning("🗑️  All collections dropped")
        else:
            logger.info("Operation cancelled")

    def close(self):
        """Close MongoDB connection."""
        self.client.close()
        logger.info("👋 MongoDB connection closed")


async def main():
    """Main setup function."""
    setup = MongoDBSetup(
        mongo_uri="mongodb://admin:admin123@localhost:27017",
        database_name="voice_agent"
    )

    try:
        # Run setup
        await setup.setup_all()

        # Uncomment to drop all collections (DANGEROUS!)
        # await setup.drop_all()

    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
    finally:
        setup.close()


if __name__ == "__main__":
    asyncio.run(main())