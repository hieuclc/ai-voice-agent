import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  PipecatClient,
  PipecatClientOptions,
  RTVIEvent,
} from '@pipecat-ai/client-js';
import { SmallWebRTCTransport } from '@pipecat-ai/small-webrtc-transport';

type ChatSession = {
  id: string;
  title?: string;
  updated_at?: string;
};

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  source: 'history' | 'live';
  timestamp?: string;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:7860';
const WEBRTC_URL =
  import.meta.env.VITE_WEBRTC_URL ?? `${API_BASE}/api/offer`;
const SESSIONS_API =
  import.meta.env.VITE_SESSIONS_API ?? `${API_BASE}/api/chat-sessions`;


const DEFAULT_ICE_SERVERS = [
  {
    urls: 'stun:ss-turn1.xirsys.com',
  },
  {
    urls: 'turn:ss-turn1.xirsys.com:3478',
    username:
      import.meta.env.VITE_TURN_USERNAME ??
      'ynuYRR28qU2zB-hB60HmYV6ulUr4Vxn3a08Fti13c0aGS-msyw6Iws7G22TbgQmgAAAAAGlBhJ9oaWV1bGNsY2JnYm4xMjM=',
    credential:
      import.meta.env.VITE_TURN_CREDENTIAL ??
      'd6c8b274-da99-11f0-ba29-0242ac140004',
  },
];

function parseSessionId(payload: unknown): string | null {
  if (!payload || typeof payload !== 'object') return null;
  const candidate =
    (payload as Record<string, unknown>).session_id ??
    (payload as Record<string, unknown>).sessionId ??
    (payload as Record<string, unknown>).id ??
    (payload as Record<string, unknown>).uuid;
  return candidate && typeof candidate === 'string'
    ? candidate
    : candidate
        ? String(candidate)
        : null;
}

function buildWebrtcUrl(sessionId: string): string {
  try {
    const url = new URL(WEBRTC_URL);
    url.searchParams.set('session_id', sessionId);
    return url.toString();
  } catch {
    return `${WEBRTC_URL}?session_id=${encodeURIComponent(sessionId)}`;
  }
}

const randomId = () =>
  typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : `session-${Date.now()}`;

const ICE_SERVERS = DEFAULT_ICE_SERVERS;

export default function App() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState('Disconnected');
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(
    null,
  );
  const [isConnecting, setIsConnecting] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const clientRef = useRef<PipecatClient | null>(null);
  const transportRef = useRef<SmallWebRTCTransport | null>(null);

  const apiHeaders = useMemo(
    () => ({
      Accept: 'application/json',
      'Content-Type': 'application/json',
    }),
    [],
  );

  const pushMessage = useCallback((msg: Omit<ChatMessage, 'id'>) => {
    console.log(msg)
    setMessages((prev) => [
      ...prev,
      { ...msg, id: `${Date.now()}-${Math.random().toString(16).slice(2)}` },
    ]);
  }, []);

  const fetchSessions = useCallback(async () => {
    setLoadingSessions(true);
    console.log("loading sessions")
    try {
      const response = await fetch(SESSIONS_API, { headers: apiHeaders });
      console.log(response)
      if (!response.ok) {
        throw new Error(`Failed to load sessions (${response.status})`);
      }
      const chat_sessions_data = await response.json();
      const data = chat_sessions_data["chat_sessions"]
      // console.log()
      if (!Array.isArray(data)) return;
      const normalized = data
        .map((item: unknown) => {
          const id = parseSessionId(item);
          if (!id) return null;
          const typed = item as Record<string, unknown>;
          return {
            id,
            title:
              (typed.title as string | undefined) ??
              (typed.name as string | undefined),
            updated_at: typed.updated_at as string | undefined,
          };
        })
        .filter(Boolean) as ChatSession[];
      setSessions(normalized);
    } catch (err) {
      console.error(err);
      setError((err as Error).message);
    } finally {
      setLoadingSessions(false);
    }
  }, [apiHeaders]);

  const fetchHistory = useCallback(
    async (sessionId: string) => {
      setLoadingHistory(true);
      try {
        const response = await fetch(`${SESSIONS_API}/${sessionId}`, {
          headers: apiHeaders,
        });
        if (!response.ok) {
          // 404 means new session with no history - that's fine
          if (response.status === 404) {
            setMessages([]);
            return;
          }
          throw new Error(`Failed to load chat history (${response.status})`);
      }
        const data = await response.json();
        const payload =
          Array.isArray(data?.messages) && data.messages.length
            ? data.messages
            : Array.isArray(data)
              ? data
              : [];

        const hydrated: ChatMessage[] = payload
          .map((item: unknown) => {
            if (!item || typeof item !== 'object') return null;
            const role = (item as Record<string, unknown>).role;
            const content = (item as Record<string, unknown>).content;
            if (
              (role === 'user' || role === 'assistant') &&
              typeof content === 'string'
            ) {
              return {
                id: `${sessionId}-${Math.random().toString(16).slice(2)}`,
                role,
                text: content,
                source: 'history' as const,
                timestamp: (item as Record<string, unknown>).timestamp as
                  | string
                  | undefined,
              };
            }
            return null;
          })
          .filter(Boolean) as ChatMessage[];

        setMessages(hydrated);
      } catch (err) {
        console.error(err);
        setError((err as Error).message);
        setMessages([]);
      } finally {
        setLoadingHistory(false);
      }
    },
    [apiHeaders],
  );

  const createNewSession = useCallback(async () => {
    try {
      const response = await fetch(`${SESSIONS_API}/create`, {
        method: 'POST',
        headers: apiHeaders,
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        throw new Error(`Failed to create session (${response.status})`);
      }
      const data = await response.json();
      const newId = data["session_id"];
      setSelectedSessionId(newId);
      setActiveSessionId(null);
      setMessages([]);
      fetchSessions();
      return newId;
    } catch (err) {
      console.error(err);
      const fallbackId = randomId();
      setSelectedSessionId(fallbackId);
      setActiveSessionId(null);
      setMessages([]);
      setError(
        `Using local session id (${(err as Error).message ?? 'unknown error'})`,
      );
      return fallbackId;
    }
  }, [apiHeaders, fetchSessions]);

  const setupAudioTrack = useCallback((track: MediaStreamTrack) => {
    if (!audioRef.current) return;
    const existing =
      audioRef.current.srcObject instanceof MediaStream
        ? audioRef.current.srcObject.getAudioTracks()[0]
        : null;

    if (existing?.id === track.id) return;
    audioRef.current.srcObject = new MediaStream([track]);
  }, []);

  const setupTrackListeners = useCallback(
    (client: PipecatClient) => {
      client.on(RTVIEvent.TrackStarted, (track, participant) => {
        if (!participant?.local && track.kind === 'audio') {
          setupAudioTrack(track);
        }
      });

      client.on(RTVIEvent.TrackStopped, (track, participant) => {
        console.debug(
          `Track stopped: ${track.kind} from ${participant?.name || 'remote'}`,
        );
      });
    },
    [setupAudioTrack],
  );

  const disconnect = useCallback(async () => {
    setIsConnecting(false);
    try {
      await clientRef.current?.disconnect();
      const stream = audioRef.current?.srcObject as MediaStream | null;
      stream?.getTracks().forEach((t) => t.stop());
      if (audioRef.current) {
        audioRef.current.srcObject = null;
      }
    } catch (err) {
      console.error(err);
      setError((err as Error).message);
    } finally {
      setIsConnected(false);
      setStatus('Disconnected');
      clientRef.current = null;
      transportRef.current = null;
    }
  }, []);

  const connect = useCallback(async () => {
    if (isConnecting || isConnected) return;
    setError(null);
    setIsConnecting(true);
    setStatus('Connecting...');
    
    const sessionId = selectedSessionId ?? activeSessionId

    // const sessionId = "test"

    console.log("sessionId", sessionId)

    try {
      if (!sessionId) throw new Error('Missing session id');

      const transport = new SmallWebRTCTransport({
        webrtcUrl: buildWebrtcUrl(sessionId),
        // webrtcUrl: "http://localhost:7860/api/offer",
        iceServers: ICE_SERVERS,
      });

      const callbacks: PipecatClientOptions['callbacks'] = {
        onConnected: () => {
          console.log('Connected successfully');
          setStatus('Connected');
          setIsConnected(true);
          // setActiveSessionId(sessionId);
        },
        onDisconnected: () => {
          console.log('Disconnected');
          setStatus('Disconnected');
          setIsConnected(false);
        },
        onBotReady: (data) => {
          console.log('Bot ready', JSON.stringify(data));
          const tracks = clientRef.current?.tracks();
          if (tracks?.bot?.audio) {
            setupAudioTrack(tracks.bot.audio);
          }
        },
        onUserTranscript: (data) => {
          if (data.final) {
            console.log(`User: ${data.text}`);
            pushMessage({
              role: 'user',
              text: data.text,
              source: 'live',
              timestamp: data.timestamp,
            });
          }
        },
        onBotTranscript: (data) => {
          if (data.text) {
            pushMessage({
              role: 'assistant',
              text: data.text,
              source: 'live',
              timestamp: data.timestamp,
            });
          }
        },
        onMessageError: (err) => {
          console.log('Message error', err);
          setError('Message error (see console for details)');
        },
        onError: (err) => {
          console.log(err);
          const errorMessage =
            err instanceof Error
              ? err.message
              : typeof err === 'string'
                ? err
                : 'Unknown error occurred';
          setError(errorMessage);
        },
      };

      const client = new PipecatClient({
        transport,
        enableMic: true,
        enableCam: false,
        callbacks,
      });

      clientRef.current = client;
      transportRef.current = transport;
      setupTrackListeners(client);
      console.log('Initializing devices...');
      await client.initDevices();
      console.log('Connecting to WebRTC...');
      await client.connect();
      console.log('Connection process completed');
    } catch (err) {
      console.error(err);
      setError((err as Error).message);
      await disconnect();
    } finally {
      setIsConnecting(false);
    }
  }, [
    activeSessionId,
    createNewSession,
    disconnect,
    isConnected,
    isConnecting,
    pushMessage,
    selectedSessionId,
    setupAudioTrack,
    setupTrackListeners,
  ]);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  useEffect(() => {
    const exists = sessions.some(s => s.id === selectedSessionId);
    console.log(selectedSessionId, exists)

    if (selectedSessionId && exists) {
      // Always fetch history - if it's a new session, API will return nothing
      fetchHistory(selectedSessionId);
    }
  }, [fetchHistory, selectedSessionId]);

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  const handleSessionSelect = (sessionId: string) => {
    setSelectedSessionId(sessionId);
    setActiveSessionId(null);
    setMessages([]);
  };

  const sessionLabel = (session: ChatSession) =>
    session.title ??
    `Session ${session.id.slice(0, 6)}${
      session.updated_at ? ` · ${new Date(session.updated_at).toLocaleString()}` : ''
    }`;

  return (
    <div className="page">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div>
            <p className="eyebrow">Voice Agent</p>
            <h1>Conversations</h1>
          </div>
          <button
            className="primary"
            onClick={createNewSession}
            disabled={isConnecting}
          >
            New session
          </button>
        </div>

        <div className="sidebar-section">
          <div className="section-heading">
            <span>Previous sessions</span>
            {loadingSessions && <span className="pill">Loading...</span>}
          </div>
          <div className="session-list">
            {sessions.length === 0 && (
              <div className="empty">No sessions yet</div>
            )}
            {sessions.map((session) => (
              <button
                key={session.id}
                className={`session-item ${
                  selectedSessionId === session.id ? 'active' : ''
                }`}
                onClick={() => handleSessionSelect(session.id)}
                disabled={isConnecting}
              >
                <div className="session-title">{sessionLabel(session)}</div>
                {session.updated_at && (
                  <div className="session-meta">
                    Updated {new Date(session.updated_at).toLocaleString()}
                  </div>
                )}
              </button>
            ))}
          </div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <p className="eyebrow">Status</p>
            <div className="status-line">
              <span className={`dot ${isConnected ? 'on' : 'off'}`} />
              {status}
            </div>
            <div className="sub">
              {selectedSessionId
                ? `Session: ${selectedSessionId}`
                : 'No session selected'}
            </div>
          </div>
          <div className="actions">
            <button
              onClick={connect}
              className="primary"
              disabled={isConnected || isConnecting}
            >
              {isConnecting ? 'Connecting…' : 'Connect'}
            </button>
            <button onClick={disconnect} disabled={!isConnected && !isConnecting}>
              Disconnect
            </button>
          </div>
        </header>

        <section className="callouts">
          {/* <div className="pill secondary">
            Voice-only. Use your microphone; the bot will speak back.
          </div> */}
          <div className="pill secondary">
            WebRTC URL: {buildWebrtcUrl(selectedSessionId ?? 'new')}
          </div>
        </section>

        <section className="chat-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Chat history</p>
              <h2>
                {selectedSessionId
                  ? 'Session timeline'
                  : 'Pick or create a session to view history'}
              </h2>
            </div>
            {loadingHistory && <span className="pill">Loading history…</span>}
          </div>

          <div className="messages" id="messages">
            {messages.length === 0 && (
              <div className="empty">No messages yet</div>
            )}
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`message ${msg.role}`}
                title={msg.timestamp ?? ''}
              >
                <div className="message-role">
                  {msg.role === 'assistant' ? 'Agent' : 'You'}
                </div>
                <div className="message-body">{msg.text}</div>
                <div className="message-meta">
                  {msg.timestamp
                    ? new Date(msg.timestamp).toLocaleTimeString()
                    : msg.source === 'history'
                      ? 'History'
                      : 'Live'}
                </div>
              </div>
            ))}
          </div>
        </section>

        {error && <div className="error-banner">Error: {error}</div>}

        <audio ref={audioRef} autoPlay />
      </main>
    </div>
  );
}
