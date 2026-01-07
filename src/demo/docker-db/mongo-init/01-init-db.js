// mongo-init/01-init-db.js
// Script này sẽ tự động chạy khi MongoDB container khởi động lần đầu

db = db.getSiblingDB('voice_agent');

// Tạo collection transcripts với validation schema
db.createCollection('transcripts', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['session_id', 'messages', 'updated_at'],
      properties: {
        session_id: {
          bsonType: 'string',
          description: 'Unique session identifier - required'
        },
        messages: {
          bsonType: 'array',
          description: 'Array of conversation messages - required',
          items: {
            bsonType: 'object',
            required: ['role', 'content'],
            properties: {
              role: {
                bsonType: 'string',
                enum: ['user', 'assistant', 'system'],
                description: 'Message role - must be user, assistant, or system'
              },
              content: {
                bsonType: 'string',
                description: 'Message content - required'
              },
              timestamp: {
                bsonType: 'string',
                description: 'ISO timestamp of the message'
              }
            }
          }
        },
        updated_at: {
          bsonType: 'date',
          description: 'Last update timestamp - required'
        },
        created_at: {
          bsonType: 'date',
          description: 'Session creation timestamp'
        },
        metadata: {
          bsonType: 'object',
          description: 'Additional session metadata',
          properties: {
            user_id: {
              bsonType: 'string',
              description: 'User identifier'
            },
            language: {
              bsonType: 'string',
              description: 'Conversation language'
            },
            total_duration: {
              bsonType: 'int',
              description: 'Total conversation duration in seconds'
            }
          }
        }
      }
    }
  }
});

// Tạo indexes để tối ưu query
db.transcripts.createIndex({ session_id: 1 }, { unique: true });
db.transcripts.createIndex({ updated_at: -1 });
db.transcripts.createIndex({ 'metadata.user_id': 1 });
db.transcripts.createIndex({ created_at: -1 });

// Tạo collection cho analytics (optional)
db.createCollection('analytics', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['session_id', 'event_type', 'timestamp'],
      properties: {
        session_id: {
          bsonType: 'string',
          description: 'Related session ID'
        },
        event_type: {
          bsonType: 'string',
          enum: ['session_start', 'session_end', 'error', 'user_feedback'],
          description: 'Type of analytics event'
        },
        timestamp: {
          bsonType: 'date',
          description: 'Event timestamp'
        },
        data: {
          bsonType: 'object',
          description: 'Event-specific data'
        }
      }
    }
  }
});

db.analytics.createIndex({ session_id: 1 });
db.analytics.createIndex({ event_type: 1 });
db.analytics.createIndex({ timestamp: -1 });

// Insert sample data để test
db.transcripts.insertOne({
  session_id: 'sample_session_001',
  messages: [
    {
      role: 'user',
      content: 'Xin chào, bạn có thể giúp tôi không?',
      timestamp: new Date().toISOString()
    },
    {
      role: 'assistant',
      content: 'Chào bạn! Tôi có thể giúp gì cho bạn?',
      timestamp: new Date().toISOString()
    }
  ],
  created_at: new Date(),
  updated_at: new Date(),
  metadata: {
    user_id: 'user_123',
    language: 'vi',
    total_duration: 0
  }
});

print('✅ Database voice_agent initialized successfully');
print('✅ Collection "transcripts" created with validation schema');
print('✅ Collection "analytics" created');
print('✅ Indexes created');
print('✅ Sample data inserted');do