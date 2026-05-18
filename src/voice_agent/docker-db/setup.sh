#!/usr/bin/env bash
set -e

# =========================
# CONFIG
# =========================
MONGO_VERSION=7.0
ADMIN_USER=admin
ADMIN_PASS=${MONGO_ADMIN_PASSWORD:-admin123}
DB_NAME=voice_agent
DATA_DIR=/data/mongodb
LOG_DIR=/data/mongodb_logs

echo "🚀 Setting up MongoDB inside container (no Docker, no SSH)"

# =========================
# STEP 1: Install MongoDB
# =========================
if ! command -v mongod >/dev/null 2>&1; then
  echo "📦 Installing MongoDB $MONGO_VERSION..."
  apt update
  apt install -y gnupg curl ca-certificates

  curl -fsSL https://pgp.mongodb.com/server-${MONGO_VERSION}.asc \
    | gpg --dearmor -o /usr/share/keyrings/mongodb-server.gpg

  echo "deb [ signed-by=/usr/share/keyrings/mongodb-server.gpg ] \
https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/${MONGO_VERSION} multiverse" \
    > /etc/apt/sources.list.d/mongodb-org.list

  apt update
  apt install -y mongodb-org
fi

# =========================
# STEP 2: Prepare folders
# =========================
mkdir -p $DATA_DIR $LOG_DIR
chown -R root:root $DATA_DIR $LOG_DIR

# =========================
# STEP 3: Start MongoDB (no auth)
# =========================
echo "▶️ Starting mongod (no auth, first boot)..."

mongod \
  --dbpath $DATA_DIR \
  --logpath $LOG_DIR/mongod.log \
  --bind_ip 127.0.0.1 \
  --fork

sleep 3

# =========================
# STEP 4: Create admin user
# =========================
echo "👤 Creating admin user..."

mongosh <<EOF
use admin
if (!db.getUser("$ADMIN_USER")) {
  db.createUser({
    user: "$ADMIN_USER",
    pwd: "$ADMIN_PASS",
    roles: [{ role: "root", db: "admin" }]
  })
  print("✅ Admin user created")
} else {
  print("ℹ️ Admin user already exists")
}
EOF

# =========================
# STEP 5: Restart with auth
# =========================
echo "🔐 Restarting MongoDB with authentication..."

mongod --shutdown --dbpath $DATA_DIR

mongod \
  --auth \
  --dbpath $DATA_DIR \
  --logpath $LOG_DIR/mongod.log \
  --bind_ip 127.0.0.1 \
  --fork

sleep 3

# =========================
# STEP 6: Create database
# =========================
mongosh \
  -u "$ADMIN_USER" \
  -p "$ADMIN_PASS" \
  --authenticationDatabase admin <<EOF
use $DB_NAME
db.createCollection("init")
print("📦 Database $DB_NAME ready")
EOF

# =========================
# STEP 7: Run schema setup
# =========================
echo "🧠 Running setup_mongodb.py..."

export MONGO_URI="mongodb://$ADMIN_USER:$ADMIN_PASS@localhost:27017/$DB_NAME?authSource=admin"

python setup_mongodb.py

echo "✨ MongoDB fully ready inside container"
