// src/pages/LiveFeed.jsx
import { useState, useEffect, useCallback, useRef } from 'react'
import toast from 'react-hot-toast'
import { useWS } from '../context/WebSocketContext'
import { getIncidentMeta, formatTime } from '../utils/incidentMeta'

const STREAM_URL = '/api/camera/stream'
const POLL_MS = 1000

// Subtle alert sound synthesizer using Web Audio API
function playAlertChime() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext
    if (!AudioCtx) return
    const ctx = new AudioCtx()
    const osc = ctx.createOscillator()
    const gain = ctx.createGain()

    osc.type = 'sawtooth'
    osc.frequency.setValueAtTime(880, ctx.currentTime) // A5
    osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.35)

    gain.gain.setValueAtTime(0.25, ctx.currentTime)
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.35)

    osc.connect(gain)
    gain.connect(ctx.destination)

    osc.start()
    osc.stop(ctx.currentTime + 0.35)
  } catch (_) {}
}

const isVideoAudioModality = (modality) => {
  if (!modality) return false
  const m = String(modality).toUpperCase()
  return (
    m.includes('V+A') ||
    m.includes('MULTIMODAL') ||
    m.includes('VIDEO+AUDIO') ||
    m.includes('VIDEO + AUDIO') ||
    (m.includes('VIDEO') && m.includes('AUDIO'))
  )
}

export default function LiveFeed() {
  const { incidents, lastEvent, connected } = useWS()
  const [highConfAlert, setHighConfAlert] = useState(null)
  const [ntfyTopic, setNtfyTopic] = useState('cctv_multimodal_incidents')
  const [testingNtfy, setTestingNtfy] = useState(false)
  const alertedKeysRef = useRef(new Set())
  const [camStatus, setCamStatus] = useState({
    running: false,
    frame_count: 0,
    fps: 0,
    uptime_sec: 0,
    active_owners_count: 0,
    auth_bypass: false,
    authorized_users: [],
  })

  // Poll camera status
  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/camera/status')
      if (res.ok) setCamStatus(await res.json())
    } catch (_) {}
  }, [])

  useEffect(() => {
    fetch('/api/notifications/topic')
      .then(r => r.json())
      .then(d => { if (d.topic) setNtfyTopic(d.topic) })
      .catch(() => {})
  }, [])

  useEffect(() => {
    fetchStatus()
    const iv = setInterval(fetchStatus, POLL_MS)
    return () => clearInterval(iv)
  }, [fetchStatus])

  // ── Automatic Delivery to ntfy: Decision >= 70% ONLY for Video+Audio, else >= 80% ───────
  useEffect(() => {
    if (!lastEvent || lastEvent.label === 'Normal' || lastEvent.auth_bypass) return
    const conf = Number(lastEvent.confidence) || 0
    // Condition of Decision: use decision value (e.g. 75), or compute from confidence * 100
    const decision = Number(
      lastEvent.decision !== undefined && lastEvent.decision !== null
        ? lastEvent.decision
        : conf * 100
    )

    // Check modality: Decision >= 70% is ONLY for Video+Audio (multimodal); single modality requires >= 80%
    const modality = lastEvent.modality_status || ''
    const isVA = isVideoAudioModality(modality)
    const requiredThreshold = isVA ? 70.0 : 80.0

    if (decision >= requiredThreshold) {
      const eventKey = `${lastEvent.incident_id || lastEvent.timestamp}-${lastEvent.label}`
      if (!alertedKeysRef.current.has(eventKey)) {
        alertedKeysRef.current.add(eventKey)

        // Trigger audible chime and visual banner
        playAlertChime()
        setHighConfAlert({
          label: lastEvent.label,
          decision: decision,
          confidence: conf,
          modality: lastEvent.modality_status,
          timestamp: lastEvent.timestamp,
          isVA: isVA,
          threshold: requiredThreshold,
        })

        // Check if backend already confirmed ntfy delivery
        const alreadyDispatched = lastEvent.alert_statuses?.some(
          a => a.channel === 'ntfy' && a.status === 'sent'
        )

        const prefix = isVA ? 'V+A DECISION >= 70%' : 'DECISION >= 80%'

        if (!alreadyDispatched) {
          // Deliver notification to ntfy user via backend API
          fetch('/api/notifications/send', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              topic: ntfyTopic,
              title: `🚨 [${prefix}] ${lastEvent.label.toUpperCase()} (${decision.toFixed(0)}%)`,
              message: `Decision: ${lastEvent.label} (${decision.toFixed(0)}%) detected in Live Feed!\nModality: ${lastEvent.modality_status || (isVA ? 'Video + Audio' : 'Video')}\nTime: ${formatTime(lastEvent.timestamp)}`,
              priority: 'urgent',
              tags: 'warning,rotating_light,cctv',
              snapshot_b64: lastEvent.snapshot_b64 || null,
            }),
          }).catch(() => {
            // Direct browser push fallback to ntfy.sh via JSON publish (avoids ISO-8859-1 header errors)
            fetch('https://ntfy.sh', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                topic: ntfyTopic,
                title: `🚨 [${prefix}] ${lastEvent.label.toUpperCase()} (${decision.toFixed(0)}%)`,
                message: `Decision: ${lastEvent.label} (${decision.toFixed(0)}%) in Live Feed!\nModality: ${lastEvent.modality_status || (isVA ? 'Video + Audio' : 'Video')}`,
                priority: 5,
                tags: ['warning', 'rotating_light', 'cctv'],
              }),
            }).catch(() => {})
          })
        }

        toast.success(
          `🚨 ${prefix}: ${lastEvent.label} (${decision.toFixed(0)}%) delivered to ntfy @${ntfyTopic}!`,
          { className: 'custom-toast', icon: '🔔', duration: 5000 }
        )
      }
    }
  }, [lastEvent, ntfyTopic])

  const handleTestNtfy = async () => {
    setTestingNtfy(true)
    try {
      const res = await fetch('/api/notifications/test-ntfy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: ntfyTopic }),
      })
      if (res.ok) {
        toast.success(`Test alert pushed to ntfy.sh/${ntfyTopic}!`, { className: 'custom-toast' })
      } else {
        toast.error('Failed to deliver test alert', { className: 'custom-toast' })
      }
    } catch {
      toast.error('Network error triggering alert', { className: 'custom-toast' })
    }
    setTestingNtfy(false)
  }

  const recentIncidents = incidents.filter(e => e.label !== 'Normal').slice(0, 10)
  const latestIncident = recentIncidents[0]
  const isStarting = !camStatus.running && camStatus.frame_count === 0 && camStatus.uptime_sec === 0

  return (
    <div>
      <div className="page-header">
        <div className="page-title-group">
          <h1>Live Feed</h1>
          <p>Real-time annotated camera stream with live incident detection &amp; ntfy.sh alerts</p>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
          {/* ntfy push indicator badge */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            background: 'rgba(6,182,212,0.12)', border: '1px solid rgba(6,182,212,0.3)',
            borderRadius: 8, padding: '4px 10px', fontSize: '0.78rem'
          }}>
            <span style={{ fontSize: '0.9rem' }}>🔔</span>
            <span style={{ color: 'var(--text-secondary)' }}>ntfy:</span>
            <a
              href={`https://ntfy.sh/${ntfyTopic}`}
              target="_blank"
              rel="noreferrer"
              style={{ color: '#38bdf8', fontFamily: 'var(--font-mono)', fontWeight: 600 }}
              title="Open ntfy topic in new tab"
            >
              @{ntfyTopic}
            </a>
            <span style={{ color: '#f87171', fontSize: '0.72rem', fontWeight: 700, marginLeft: 2 }}>(V+A Decision &ge; 70% auto)</span>
            <button
              onClick={handleTestNtfy}
              disabled={testingNtfy}
              style={{
                marginLeft: 4, background: 'rgba(6,182,212,0.25)', border: 'none',
                borderRadius: 4, color: '#38bdf8', cursor: 'pointer',
                fontSize: '0.7rem', padding: '2px 6px'
              }}
            >
              {testingNtfy ? '…' : 'Test'}
            </button>
          </div>

          {camStatus.running ? (
            <span className="badge badge-active">● LIVE — {camStatus.fps} FPS</span>
          ) : (
            <span className="badge badge-inactive">Camera Off</span>
          )}
          <span className={`badge ${connected ? 'badge-active' : 'badge-inactive'}`}>
            WS {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      <div className="two-col" style={{ gap: 24, alignItems: 'start' }}>
        {/* MJPEG Stream & Banners */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          
          {/* Authorized Personnel Bypass Banner */}
          {camStatus.auth_bypass && (
            <div style={{
              background: 'linear-gradient(90deg, rgba(16,185,129,0.18), rgba(6,182,212,0.18))',
              border: '1px solid rgba(16,185,129,0.4)',
              borderRadius: 10,
              padding: '12px 16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 12,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={{ fontSize: '1.6rem' }}>🛡️</span>
                <div>
                  <div style={{ fontWeight: 800, color: '#10b981', fontSize: '0.95rem' }}>
                    AUTHORIZED PERSONNEL DETECTED: {(camStatus.authorized_users || []).join(', ').toUpperCase() || 'TRUSTED USER'}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#e2e8f0', marginTop: 2 }}>
                    Detection bypass is active. Emergency alerts and false alarms are suppressed for this verified user.
                  </div>
                </div>
              </div>
              <span className="badge" style={{ background: '#10b981', color: '#000', fontWeight: 700 }}>
                ● BYPASS ACTIVE
              </span>
            </div>
          )}

          {/* Decision Alert Banner (Decision >= 70% for V+A, >= 80% otherwise) */}
          {highConfAlert && (
            <div className="high-conf-alert-banner">
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={{ fontSize: '1.6rem' }}>🚨</span>
                <div>
                  <div style={{ fontWeight: 800, color: '#f87171', fontSize: '0.95rem' }}>
                    {highConfAlert.isVA ? 'V+A DECISION ≥ 70%' : 'DECISION ≥ 80%'}: {highConfAlert.label.toUpperCase()} ({highConfAlert.decision?.toFixed(0) || (highConfAlert.confidence * 100).toFixed(0)}%)
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#e2e8f0', marginTop: 2 }}>
                    Alert dispatched to ntfy.sh topic <code style={{ color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>@{ntfyTopic}</code> · {highConfAlert.modality || (highConfAlert.isVA ? 'Video + Audio' : 'Video')} · {formatTime(highConfAlert.timestamp)}
                  </div>
                </div>
              </div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <a
                  href={`https://ntfy.sh/${ntfyTopic}`}
                  target="_blank"
                  rel="noreferrer"
                  className="btn btn-ghost btn-sm"
                  style={{ border: '1px solid rgba(255,255,255,0.2)', fontSize: '0.75rem' }}
                >
                  View on ntfy ↗
                </a>
                <button
                  className="btn btn-ghost btn-sm"
                  onClick={() => setHighConfAlert(null)}
                  style={{ opacity: 0.8 }}
                >
                  ✕
                </button>
              </div>
            </div>
          )}

          <div className="stream-panel">
            {camStatus.running ? (
              <>
                <img
                  src={STREAM_URL}
                  alt="Live CCTV Feed"
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                />
                <div className="stream-overlay">
                  <div className="stream-badge">
                    <div className="dot" />
                    LIVE
                  </div>
                  {latestIncident && (
                    <span style={{ fontSize: '0.75rem', color: '#fff', opacity: 0.85 }}>
                      {getIncidentMeta(latestIncident.label).emoji} {latestIncident.label} — {(latestIncident.confidence * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
              </>
            ) : isStarting ? (
              /* Model warm-up skeleton */
              <div style={{
                display: 'flex', flexDirection: 'column', alignItems: 'center',
                justifyContent: 'center', height: '100%', gap: 16, padding: 40,
              }}>
                <div style={{
                  width: 44, height: 44, borderRadius: '50%',
                  border: '3px solid var(--border)',
                  borderTopColor: 'var(--accent)',
                  animation: 'spin 0.9s linear infinite',
                }} />
                <div style={{ textAlign: 'center' }}>
                  <p style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>
                    Initializing pipeline…
                  </p>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    Stream will appear once models are ready
                  </p>
                </div>
              </div>
            ) : (
              <div style={{
                display: 'flex', flexDirection: 'column', alignItems: 'center',
                justifyContent: 'center', height: '100%', gap: 12, padding: 40,
              }}>
                <span style={{ fontSize: '3rem' }}>📷</span>
                <p style={{ color: 'var(--text-muted)', textAlign: 'center' }}>
                  Camera is not running.<br />Go to Dashboard to start analysis.
                </p>
              </div>
            )}
          </div>

          {/* Camera info row */}
          {camStatus.running && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
              {[
                ['FPS', camStatus.fps],
                ['Frames', camStatus.frame_count?.toLocaleString()],
                ['Uptime', `${Math.floor(camStatus.uptime_sec)}s`],
              ].map(([k, v]) => (
                <div key={k} style={{
                  background: 'var(--bg-glass)', border: '1px solid var(--border)',
                  borderRadius: 8, padding: '10px 14px', textAlign: 'center',
                }}>
                  <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{k}</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 700, fontFamily: 'var(--font-mono)', marginTop: 2 }}>{v}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Live Event Ticker */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div className="card-header">
            <span className="card-title">Incident Events</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Latest 10</span>
          </div>
          {recentIncidents.length === 0 ? (
            <div className="empty-state">
              <span style={{ fontSize: '2.5rem' }}>✅</span>
              <p>No incidents detected</p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {recentIncidents.map((ev, i) => {
                const meta = getIncidentMeta(ev.label)
                const decisionVal = Number(
                  ev.decision !== undefined && ev.decision !== null
                    ? ev.decision
                    : (Number(ev.confidence || 0) * 100)
                )
                const modalityStr = String(ev.modality_status || '').toUpperCase()
                const isVA = isVideoAudioModality(modalityStr)
                const reqThreshold = isVA ? 70.0 : 80.0
                const isHighDecision = decisionVal >= reqThreshold
                return (
                  <div key={i} style={{
                    display: 'flex', gap: 12, alignItems: 'center',
                    padding: '10px 12px',
                    background: isHighDecision ? 'rgba(239,68,68,0.08)' : 'var(--bg-glass)',
                    borderRadius: 8,
                    borderLeft: `3px solid ${meta.color}`,
                    border: isHighDecision ? '1px solid rgba(239,68,68,0.3)' : '1px solid var(--border)',
                    borderLeftWidth: 4,
                    borderLeftColor: meta.color,
                  }}>
                    <span style={{ fontSize: '1.2rem' }}>{meta.emoji}</span>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ fontSize: '0.875rem', fontWeight: 700, color: meta.color }}>{ev.label}</span>
                        {isHighDecision && (
                          <span className="badge badge-ntfy" style={{ fontSize: '0.62rem', padding: '1px 6px' }}>
                            🔔 {isVA ? 'V+A DECISION ≥ 70%' : 'DECISION ≥ 80%'}
                          </span>
                        )}
                      </div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>
                        Decision: {decisionVal.toFixed(0)}% · {ev.modality_status} · {ev.owners_notified} notified
                      </div>
                    </div>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      {formatTime(ev.timestamp)}
                    </span>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
