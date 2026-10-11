// src/pages/Dashboard.jsx
import { useEffect, useState, useRef, useCallback } from 'react'
import { useWS } from '../context/WebSocketContext'
import { getIncidentMeta, getSeverityClass, formatTime } from '../utils/incidentMeta'

const API = '/api'

// ─── Phenomenon mode definitions ─────────────────────────────────────────────
const MODES = [
  {
    id: 'multimodal',
    label: '🔀 Multimodal',
    desc: 'Video + Audio fusion',
    config: { enable_yolo: true, enable_video_incident: true, enable_audio_incident: true },
  },
  {
    id: 'video',
    label: '🎬 Video Only',
    desc: 'GRU-ResNet + YOLO, no audio',
    config: { enable_yolo: true, enable_video_incident: true, enable_audio_incident: false },
  },
  {
    id: 'audio',
    label: '🎙️ Audio Only',
    desc: 'Keras audio model, no video inference',
    config: { enable_yolo: false, enable_video_incident: false, enable_audio_incident: true },
  },
  {
    id: 'custom',
    label: '⚙️ Custom',
    desc: 'Configure manually',
    config: null,
  },
]

// ─── Sub-components ───────────────────────────────────────────────────────────
function StatCard({ icon, iconBg, value, label }) {
  return (
    <div className="stat-card">
      <div className="stat-icon" style={{ background: iconBg }}>{icon}</div>
      <div className="stat-info">
        <h3>{value}</h3>
        <p>{label}</p>
      </div>
    </div>
  )
}

function IncidentRow({ ev }) {
  const meta = getIncidentMeta(ev.label)
  return (
    <div className={`incident-card ${getSeverityClass(ev.label)}`}>
      <div className="incident-icon" style={{ background: `${meta.color}22` }}>
        <span style={{ fontSize: '1.1rem' }}>{meta.emoji}</span>
      </div>
      <div className="incident-body">
        <div className="incident-title" style={{ color: meta.color }}>{ev.label}</div>
        <div className="conf-bar">
          <div className="conf-fill" style={{ width: `${ev.confidence * 100}%`, background: meta.color }} />
        </div>
        <div className="incident-meta" style={{ marginTop: 6 }}>
          <span>{(ev.confidence * 100).toFixed(1)}% confidence</span>
          <span>·</span>
          <span>{ev.modality_status}</span>
          <span>·</span>
          <span>{ev.owners_notified} notified</span>
          <span>·</span>
          <span>{formatTime(ev.timestamp)}</span>
        </div>
        {ev.alert_statuses?.length > 0 && (
          <div className="incident-meta" style={{ marginTop: 4 }}>
            {ev.alert_statuses.map((a, i) => (
              <span key={i} className={`badge badge-${a.status}`} style={{ fontSize: '0.65rem' }}>
                {a.channel?.toUpperCase()} → {a.owner} [{a.status}]
              </span>
            ))}
          </div>
        )}
      </div>
      {ev.snapshot_b64 && (
        <img
          src={`data:image/jpeg;base64,${ev.snapshot_b64}`}
          alt="snapshot"
          style={{ width: 70, height: 50, objectFit: 'cover', borderRadius: 6, flexShrink: 0 }}
        />
      )}
    </div>
  )
}

// ─── Pill-mode selector ───────────────────────────────────────────────────────
function ModePills({ selected, onChange }) {
  return (
    <div style={{
      display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16,
      background: 'var(--bg-glass)', borderRadius: 12, padding: 6,
      border: '1px solid var(--border)',
    }}>
      {MODES.map(m => (
        <button
          key={m.id}
          onClick={() => onChange(m.id)}
          title={m.desc}
          style={{
            flex: 1, minWidth: 100,
            padding: '8px 12px',
            borderRadius: 8,
            border: 'none',
            cursor: 'pointer',
            fontSize: '0.8rem',
            fontWeight: 600,
            transition: 'all 0.2s',
            background: selected === m.id ? 'var(--accent)' : 'transparent',
            color: selected === m.id ? '#fff' : 'var(--text-secondary)',
            boxShadow: selected === m.id ? '0 2px 8px rgba(99,102,241,0.4)' : 'none',
          }}
        >
          {m.label}
        </button>
      ))}
    </div>
  )
}

// ─── Main Dashboard ───────────────────────────────────────────────────────────
export default function Dashboard() {
  const { incidents, connected } = useWS()
  const [stats, setStats] = useState(null)
  const [camStatus, setCamStatus] = useState({ running: false })
  const [loading, setLoading] = useState(false)   // waiting for HTTP response
  const [starting, setStarting] = useState(false) // waiting for ML models to warm up
  const [pipelineError, setPipelineError] = useState(null)
  const startPollRef = useRef(null)

  // Selected phenomenon mode
  const [modeId, setModeId] = useState('multimodal')

  // Manual form values for 'custom' mode
  const [customForm, setCustomForm] = useState({
    source: '0',
    enable_yolo: true,
    enable_video_incident: true,
    enable_audio_incident: true,
    video_weight: 0.55,
    audio_weight: 0.45,
    yolo_conf: 0.35,
  })

  // Source input (shared across modes)
  const [source, setSource] = useState('0')

  // ── Unified status + stats poller ──────────────────────────────────────────
  const fetchStatus = useCallback(() => {
    fetch(`${API}/camera/status`)
      .then(r => r.json())
      .then(d => {
        setCamStatus(d)
        if (d.pipeline_error) setPipelineError(d.pipeline_error)
        else setPipelineError(null)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    fetchStatus()
    fetch(`${API}/incidents/stats`).then(r => r.json()).then(setStats).catch(() => {})

    const iv = setInterval(() => {
      fetchStatus()
      fetch(`${API}/incidents/stats`).then(r => r.json()).then(setStats).catch(() => {})
    }, 3000)               // 3s instead of 5s — more responsive FPS display

    return () => clearInterval(iv)
  }, [fetchStatus])

  // ── Startup polling — 1s fast poll until camera is confirmed running ────────
  const startStartupPoll = useCallback(() => {
    let attempts = 0
    const MAX = 60  // give up after ~60s

    const poll = () => {
      attempts++
      fetch(`${API}/camera/status`)
        .then(r => r.json())
        .then(d => {
          setCamStatus(d)
          if (d.pipeline_error) {
            setPipelineError(d.pipeline_error)
            setStarting(false)
            clearTimeout(startPollRef.current)
          } else if (d.running) {
            setStarting(false)
            clearTimeout(startPollRef.current)
          } else if (attempts < MAX) {
            startPollRef.current = setTimeout(poll, 1000)
          } else {
            setStarting(false) // timed out
          }
        })
        .catch(() => {
          if (attempts < MAX) startPollRef.current = setTimeout(poll, 1000)
          else setStarting(false)
        })
    }

    startPollRef.current = setTimeout(poll, 1000)
  }, [])

  // ── Build the payload based on selected mode ────────────────────────────────
  const buildPayload = () => {
    const mode = MODES.find(m => m.id === modeId)
    if (modeId === 'custom') {
      return { ...customForm, source }
    }
    // For preset modes: use mode config + shared source + weight defaults
    const weights = modeId === 'audio'
      ? { video_weight: 0.0, audio_weight: 1.0 }
      : modeId === 'video'
        ? { video_weight: 1.0, audio_weight: 0.0 }
        : { video_weight: 0.55, audio_weight: 0.45 }

    return {
      source,
      ...mode.config,
      ...weights,
      yolo_conf: 0.35,
    }
  }

  const handleStart = async () => {
    setLoading(true)
    setPipelineError(null)
    try {
      const res = await fetch(`${API}/camera/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildPayload()),
      })
      if (!res.ok) {
        const err = await res.json()
        setPipelineError(err.detail || 'Failed to start')
        setLoading(false)
        return
      }
      const data = await res.json()
      setCamStatus(data)
      setStarting(true)    // models are loading in background
      startStartupPoll()   // fast 1s poll until running:true
    } catch (e) {
      setPipelineError(`Network error: ${e.message}`)
    }
    setLoading(false)
  }

  const handleStop = async () => {
    setLoading(true)
    clearTimeout(startPollRef.current)
    setStarting(false)
    try {
      const res = await fetch(`${API}/camera/stop`, { method: 'POST' })
      const data = await res.json()
      setCamStatus(data)
    } catch (e) { console.error(e) }
    setLoading(false)
  }

  const incidentClasses = ['Abuse', 'CarAccident', 'Explosion', 'Fighting', 'Riot', 'Shooting']
  const activeMode = MODES.find(m => m.id === modeId)

  return (
    <div>
      <div className="page-header">
        <div className="page-title-group">
          <h1>Dashboard</h1>
          <p>Real-time multimodal CCTV incident monitoring &amp; alert center</p>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <span className={`badge ${connected ? 'badge-active' : 'badge-inactive'}`}>
            {connected ? '● Live' : '○ Offline'}
          </span>
        </div>
      </div>

      {/* Stat Cards */}
      <div className="stat-grid">
        <StatCard icon="🚨" iconBg="rgba(239,68,68,0.2)"   value={stats?.total_incidents ?? '—'} label="Total Incidents" />
        <StatCard icon="📩" iconBg="rgba(59,130,246,0.2)"  value={stats?.alerts_sent ?? '—'}     label="Alerts Sent" />
        <StatCard icon="📹" iconBg="rgba(16,185,129,0.2)"  value={camStatus.running ? `${camStatus.fps} FPS` : 'Stopped'} label="Camera Status" />
        <StatCard icon="⏱️" iconBg="rgba(168,85,247,0.2)" value={camStatus.running ? `${Math.floor(camStatus.uptime_sec)}s` : '—'} label="Uptime" />
      </div>

      <div className="two-col" style={{ gap: 24 }}>
        {/* Left: Camera Control + Today's Incidents */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

          {/* Camera Control Card */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Camera Control</span>
              {camStatus.running && (
                <span className="badge badge-active" style={{ animation: 'pulse-dot 1.5s infinite' }}>● LIVE</span>
              )}
              {starting && !camStatus.running && (
                <span className="badge badge-inactive" style={{ animation: 'pulse-dot 1.5s infinite' }}>⏳ Starting…</span>
              )}
            </div>

            {/* Pipeline error banner */}
            {pipelineError && (
              <div style={{
                background: 'rgba(239,68,68,0.12)',
                border: '1px solid rgba(239,68,68,0.4)',
                borderRadius: 8, padding: '10px 14px',
                marginBottom: 14, fontSize: '0.8rem',
                color: '#f87171', display: 'flex', gap: 8, alignItems: 'flex-start',
              }}>
                <span style={{ fontSize: '1rem', flexShrink: 0 }}>⚠️</span>
                <span><strong>Pipeline error:</strong> {pipelineError}</span>
              </div>
            )}

            {!camStatus.running && !starting ? (
              <>
                {/* Source input */}
                <div className="form-group">
                  <label className="form-label">Video Source</label>
                  <input
                    className="form-input"
                    value={source}
                    onChange={e => setSource(e.target.value)}
                    placeholder="0 (webcam) or /path/to/video.mp4"
                  />
                </div>

                {/* Phenomenon mode pills */}
                <div className="form-group" style={{ marginBottom: 8 }}>
                  <label className="form-label">Analysis Mode</label>
                  <ModePills selected={modeId} onChange={setModeId} />
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: -8, marginBottom: 0 }}>
                    {activeMode?.desc}
                  </p>
                </div>

                {/* Custom mode advanced panel */}
                {modeId === 'custom' && (
                  <div style={{
                    background: 'var(--bg-glass)', border: '1px solid var(--border)',
                    borderRadius: 10, padding: 14, marginBottom: 14,
                    display: 'flex', flexDirection: 'column', gap: 12,
                  }}>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: 0, fontWeight: 600 }}>
                      ⚙️ Advanced Configuration
                    </p>

                    {/* Toggle switches */}
                    <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                      {[
                        { key: 'enable_yolo',             label: '🔍 YOLO Detection' },
                        { key: 'enable_video_incident',   label: '🎬 Video Model' },
                        { key: 'enable_audio_incident',   label: '🎙️ Audio Model' },
                      ].map(({ key, label }) => (
                        <label key={key} style={{
                          display: 'flex', alignItems: 'center', gap: 8,
                          fontSize: '0.8rem', color: 'var(--text-secondary)',
                          cursor: 'pointer', userSelect: 'none',
                        }}>
                          <div style={{
                            width: 36, height: 20, borderRadius: 10,
                            background: customForm[key] ? 'var(--accent)' : 'var(--border)',
                            position: 'relative', transition: 'background 0.2s', cursor: 'pointer',
                          }}
                            onClick={() => setCustomForm(p => ({ ...p, [key]: !p[key] }))}
                          >
                            <div style={{
                              position: 'absolute', top: 2,
                              left: customForm[key] ? 18 : 2,
                              width: 16, height: 16, borderRadius: '50%',
                              background: '#fff', transition: 'left 0.2s',
                            }} />
                          </div>
                          {label}
                        </label>
                      ))}
                    </div>

                    {/* Weight sliders */}
                    <div className="form-row" style={{ marginBottom: 0, gap: 12 }}>
                      {[
                        { key: 'video_weight', label: 'Video Weight' },
                        { key: 'audio_weight', label: 'Audio Weight' },
                      ].map(({ key, label }) => (
                        <div key={key} className="form-group" style={{ marginBottom: 0 }}>
                          <label className="form-label">
                            {label} <span style={{ color: 'var(--accent)', fontWeight: 700 }}>{customForm[key].toFixed(2)}</span>
                          </label>
                          <input
                            type="range" step="0.05" min="0" max="1"
                            style={{ width: '100%', accentColor: 'var(--accent)' }}
                            value={customForm[key]}
                            onChange={e => setCustomForm(p => ({ ...p, [key]: +e.target.value }))}
                          />
                        </div>
                      ))}
                    </div>

                    {/* YOLO confidence */}
                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">
                        YOLO Confidence Threshold <span style={{ color: 'var(--accent)', fontWeight: 700 }}>{customForm.yolo_conf.toFixed(2)}</span>
                      </label>
                      <input
                        type="range" step="0.05" min="0.1" max="0.95"
                        style={{ width: '100%', accentColor: 'var(--accent)' }}
                        value={customForm.yolo_conf}
                        onChange={e => setCustomForm(p => ({ ...p, yolo_conf: +e.target.value }))}
                      />
                    </div>
                  </div>
                )}

                <button
                  className="btn btn-primary"
                  onClick={handleStart}
                  disabled={loading}
                  style={{ width: '100%' }}
                >
                  {loading ? 'Sending request…' : '▶ Start Analysis'}
                </button>
              </>
            ) : starting && !camStatus.running ? (
              /* ── Model warm-up skeleton ── */
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <div style={{
                  background: 'var(--bg-glass)', borderRadius: 10, padding: 20,
                  display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12,
                  border: '1px solid var(--border)',
                }}>
                  <div style={{
                    width: 44, height: 44, borderRadius: '50%',
                    border: '3px solid var(--border)',
                    borderTopColor: 'var(--accent)',
                    animation: 'spin 0.9s linear infinite',
                  }} />
                  <div style={{ textAlign: 'center' }}>
                    <p style={{ fontWeight: 700, marginBottom: 4, color: 'var(--text-primary)' }}>
                      Initializing ML models…
                    </p>
                    <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                      Loading {activeMode?.label} pipeline. This may take 10–30 seconds.
                    </p>
                  </div>
                  {/* Animated skeleton bars */}
                  <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {['GRU-ResNet Video Model', 'Keras Audio Head', 'YOLOv8 Detector'].map((name, i) => (
                      modeId !== 'audio' || name.includes('Audio') ? (
                        <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', width: 160, flexShrink: 0 }}>{name}</span>
                          <div style={{
                            flex: 1, height: 6, borderRadius: 4,
                            background: 'var(--border)', overflow: 'hidden',
                          }}>
                            <div style={{
                              height: '100%', borderRadius: 4,
                              background: 'linear-gradient(90deg, var(--accent), #818cf8)',
                              animation: `skeleton-slide 1.4s ease-in-out ${i * 0.3}s infinite`,
                              width: '60%',
                            }} />
                          </div>
                        </div>
                      ) : null
                    ))}
                  </div>
                </div>
                <button
                  className="btn btn-danger"
                  onClick={handleStop}
                  disabled={loading}
                  style={{ width: '100%' }}
                >
                  ✕ Cancel
                </button>
              </div>
            ) : (
              /* ── Running state ── */
              <div>
                <div style={{ display: 'flex', gap: 16, marginBottom: 16 }}>
                  <div style={{ flex: 1, background: 'var(--bg-glass)', borderRadius: 8, padding: '10px 14px' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 2 }}>SOURCE</div>
                    <div style={{ fontSize: '0.9rem', fontFamily: 'var(--font-mono)' }}>{camStatus.source}</div>
                  </div>
                  <div style={{ flex: 1, background: 'var(--bg-glass)', borderRadius: 8, padding: '10px 14px' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 2 }}>MODE</div>
                    <div style={{ fontSize: '0.9rem', fontFamily: 'var(--font-mono)' }}>{activeMode?.label ?? modeId}</div>
                  </div>
                  <div style={{ flex: 1, background: 'var(--bg-glass)', borderRadius: 8, padding: '10px 14px' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 2 }}>FRAMES</div>
                    <div style={{ fontSize: '0.9rem', fontFamily: 'var(--font-mono)' }}>{camStatus.frame_count?.toLocaleString()}</div>
                  </div>
                </div>
                <button
                  className="btn btn-danger"
                  onClick={handleStop}
                  disabled={loading}
                  style={{ width: '100%' }}
                >
                  {loading ? 'Stopping…' : '■ Stop Analysis'}
                </button>
              </div>
            )}
          </div>

          {/* Today's Incident Breakdown */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Today's Incidents</span>
            </div>
            <div className="class-bars">
              {incidentClasses.map(cls => {
                const count = stats?.today_by_class?.[cls] || 0
                const total = Object.values(stats?.today_by_class || {}).reduce((a, b) => a + b, 0)
                const pct = total > 0 ? (count / total * 100) : 0
                const meta = getIncidentMeta(cls)
                return (
                  <div className="class-bar-row" key={cls}>
                    <span className="class-bar-label">{cls}</span>
                    <div className="class-bar-track">
                      <div className="class-bar-fill" style={{ width: `${pct}%`, background: meta.color }} />
                    </div>
                    <span className="class-bar-pct">{count}</span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        {/* Right: Live Incident Feed */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="card-header">
            <span className="card-title">Live Incident Feed</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{incidents.length} events</span>
          </div>
          {incidents.length === 0 ? (
            <div className="empty-state">
              <span style={{ fontSize: '2.5rem' }}>👁️</span>
              <p>No incidents detected yet.<br />Start camera analysis to begin monitoring.</p>
            </div>
          ) : (
            <div className="incident-ticker">
              {incidents.filter(e => e.label !== 'Normal').map((ev, i) => (
                <IncidentRow key={`${ev.incident_id}-${i}`} ev={ev} />
              ))}
              {incidents.filter(e => e.label !== 'Normal').length === 0 && (
                <div className="empty-state" style={{ padding: '30px 0' }}>
                  <span style={{ fontSize: '2rem' }}>✅</span>
                  <p>All clear — no incidents detected.</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
