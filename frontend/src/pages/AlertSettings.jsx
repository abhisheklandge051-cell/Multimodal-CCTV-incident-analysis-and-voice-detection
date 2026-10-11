// src/pages/AlertSettings.jsx
import { useState, useEffect } from 'react'
import toast from 'react-hot-toast'

export default function AlertSettings() {
  const [owners, setOwners] = useState([])
  const [stats, setStats] = useState(null)
  const [ntfyTopic, setNtfyTopic] = useState('incident_alert_shubh')
  const [testingAlert, setTestingAlert] = useState(false)
  const [savingTopic, setSavingTopic] = useState(false)
  const [testResult, setTestResult] = useState(null)

  useEffect(() => {
    fetch('/api/owners').then(r => r.json()).then(setOwners).catch(() => {})
    fetch('/api/incidents/stats').then(r => r.json()).then(setStats).catch(() => {})

    // Load current ntfy settings from backend
    fetch('/api/notifications/settings')
      .then(r => r.json())
      .then(data => {
        if (data.topic) setNtfyTopic(data.topic)
      })
      .catch(() => {})
  }, [])

  const toggleOwner = async (o) => {
    try {
      const res = await fetch(`/api/owners/${o.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ active: !o.active }),
      })
      if (res.ok) {
        setOwners(prev => prev.map(x => x.id === o.id ? { ...x, active: !o.active } : x))
        toast.success(`${o.name} ${!o.active ? 'activated' : 'deactivated'}`, { className: 'custom-toast' })
      }
    } catch {
      toast.error('Failed to update owner status', { className: 'custom-toast' })
    }
  }

  const sanitizeTopic = (t) => {
    if (!t) return ''
    let s = t.trim()
    if (s.includes('ntfy.sh/')) {
      s = s.split('ntfy.sh/').pop()
    }
    return s.replace(/^\/+|\/+$/g, '').replace(/[^a-zA-Z0-9_-]/g, '_')
  }

  const handleSaveTopic = async () => {
    const clean = sanitizeTopic(ntfyTopic)
    if (!clean) {
      toast.error('Please enter a valid topic name (letters, numbers, underscores)', { className: 'custom-toast' })
      return
    }
    setNtfyTopic(clean)
    setSavingTopic(true)
    try {
      const res = await fetch('/api/notifications/settings/topic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: clean }),
      })
      if (res.ok) {
        const data = await res.json()
        setNtfyTopic(data.topic)
        localStorage.setItem('ntfy_topic', data.topic)
        toast.success(`ntfy topic updated to "${data.topic}"!`, { className: 'custom-toast' })
      } else {
        const err = await res.json().catch(() => ({}))
        toast.error(err.detail || 'Failed to update topic', { className: 'custom-toast' })
      }
    } catch (err) {
      toast.error(`Network error saving topic: ${err.message}`, { className: 'custom-toast' })
    }
    setSavingTopic(false)
  }

  const handleSendTestAlert = async () => {
    const clean = sanitizeTopic(ntfyTopic) || 'incident_alert_shubh'
    setNtfyTopic(clean)
    setTestingAlert(true)
    setTestResult(null)

    // 1. First attempt: call backend test endpoint (uses test_alert.py workflow)
    let backendSuccess = false
    try {
      const res = await fetch('/api/notifications/test-ntfy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: clean }),
      })

      if (res.ok) {
        const data = await res.json()
        setTestResult({ success: true, message: `Status 200: Alert sent successfully to ntfy.sh/${data.topic}!` })
        toast.success('Test alert delivered to ntfy.sh!', { className: 'custom-toast' })
        backendSuccess = true
      } else {
        const errData = await res.json().catch(() => ({}))
        console.warn('Backend test-ntfy returned error, attempting direct browser push fallback:', errData)
      }
    } catch (backendErr) {
      console.warn('Backend unavailable, attempting direct browser push fallback:', backendErr)
    }

    if (backendSuccess) {
      setTestingAlert(false)
      return
    }

    // 2. Direct browser push fallback to https://ntfy.sh via JSON API
    // (Uses JSON body instead of headers to prevent ISO-8859-1 code point errors)
    try {
      const directRes = await fetch('https://ntfy.sh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic: clean,
          title: '🚨 Suspicious Activity Detected!',
          message: 'Movement detected in Camera Feed 1 (Direct Test Alert)',
          priority: 4,
          tags: ['warning', 'rotating_light'],
        }),
      })

      if (directRes.ok) {
        setTestResult({ success: true, message: `Direct push successful to ntfy.sh/${clean}!` })
        toast.success('Direct test alert delivered to ntfy.sh!', { className: 'custom-toast' })
      } else {
        const txt = await directRes.text().catch(() => '')
        setTestResult({ success: false, message: `ntfy.sh returned status ${directRes.status}: ${txt}` })
        toast.error(`Failed to send test alert (${directRes.status})`, { className: 'custom-toast' })
      }
    } catch (directErr) {
      setTestResult({ success: false, message: `Connection error: ${directErr.message}` })
      toast.error(`Error: ${directErr.message}`, { className: 'custom-toast' })
    }

    setTestingAlert(false)
  }

  const channelEmoji = (ch) => ch === 'ntfy' ? '🔔' : ch === 'whatsapp' ? '💬' : '📱'

  const totalAlerts = stats?.alerts_sent || 0
  const totalIncidents = stats?.total_incidents || 0
  const successRate = totalIncidents > 0 ? ((totalAlerts / Math.max(totalIncidents, 1)) * 100).toFixed(1) : '—'

  return (
    <div>
      <div className="page-header">
        <div className="page-title-group">
          <h1>Settings &amp; Alerts</h1>
          <p>Configure ntfy.sh push notifications, live feed alert rules, and subscriber preferences</p>
        </div>
      </div>

      <div className="two-col" style={{ gap: 24, alignItems: 'start' }}>
        {/* Left: ntfy.sh Application Setup & Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          
          {/* ntfy.sh Configuration Card (Replaced Twilio) */}
          <div className="card" style={{ border: '1px solid rgba(6,182,212,0.35)' }}>
            <div className="card-header" style={{ borderBottom: '1px solid var(--border)', paddingBottom: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{ fontSize: '1.4rem' }}>🔔</span>
                <div>
                  <span className="card-title" style={{ fontSize: '1.05rem', color: '#38bdf8' }}>ntfy.sh Application Setup</span>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Real-time push notifications for mobile &amp; web</div>
                </div>
              </div>
              <span className="badge badge-ntfy" style={{ fontSize: '0.72rem' }}>● Push Service Active</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginTop: 16 }}>
              {/* Topic Input Row */}
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label" style={{ color: '#38bdf8' }}>ntfy Topic Name</label>
                <div style={{ display: 'flex', gap: 8 }}>
                  <input
                    className="form-input"
                    value={ntfyTopic}
                    onChange={e => setNtfyTopic(e.target.value)}
                    placeholder="e.g. incident_alert_shubh"
                    style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}
                  />
                  <button
                    className="btn btn-primary"
                    onClick={handleSaveTopic}
                    disabled={savingTopic}
                    style={{ flexShrink: 0, padding: '8px 16px' }}
                  >
                    {savingTopic ? 'Saving…' : 'Save'}
                  </button>
                </div>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  Server: <code style={{ color: 'var(--accent-cyan)' }}>https://ntfy.sh</code> · Full URL: <code style={{ color: 'var(--accent-cyan)' }}>https://ntfy.sh/{ntfyTopic}</code>
                </span>
              </div>

              {/* Automatic Live Feed Alert Rule Banner */}
              <div className="ntfy-banner">
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: '1.1rem' }}>⚡</span>
                  <strong style={{ fontSize: '0.85rem', color: 'var(--text-primary)' }}>Live Feed Automatic Notification Rule</strong>
                </div>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', margin: 0 }}>
                  Whenever the live camera feed reaches <span style={{ color: '#f87171', fontWeight: 700 }}>Decision &ge; 70% in Video+Audio multimodal analysis</span> (or &ge; 80% for single modality), an emergency push alert with snapshot, alert sound, and incident tags is immediately delivered to <span style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>@{ntfyTopic}</span>.
                </p>
              </div>

              {/* Action Buttons: Test Alert & Open Web App */}
              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                <button
                  className="btn btn-primary"
                  onClick={handleSendTestAlert}
                  disabled={testingAlert}
                  style={{
                    flex: 1, minWidth: 170, display: 'flex', alignItems: 'center',
                    justifyContent: 'center', gap: 8, background: 'linear-gradient(135deg, #0284c7, #2563eb)'
                  }}
                >
                  <span>📲</span>
                  <span>{testingAlert ? 'Sending test…' : 'Send Test Alert'}</span>
                </button>

                <a
                  href={`https://ntfy.sh/${ntfyTopic}`}
                  target="_blank"
                  rel="noreferrer"
                  className="btn btn-ghost"
                  style={{
                    flex: 1, minWidth: 160, display: 'flex', alignItems: 'center',
                    justifyContent: 'center', gap: 6, border: '1px solid var(--border)'
                  }}
                >
                  <span>🌐</span>
                  <span>Open ntfy Web Client ↗</span>
                </a>
              </div>

              {/* Test Alert Status Message */}
              {testResult && (
                <div style={{
                  background: testResult.success ? 'rgba(16,185,129,0.12)' : 'rgba(239,68,68,0.12)',
                  border: `1px solid ${testResult.success ? 'rgba(16,185,129,0.4)' : 'rgba(239,68,68,0.4)'}`,
                  borderRadius: 8, padding: '10px 14px', fontSize: '0.8rem',
                  color: testResult.success ? '#34d399' : '#f87171',
                  display: 'flex', alignItems: 'center', gap: 8
                }}>
                  <span>{testResult.success ? '✅' : '❌'}</span>
                  <span>{testResult.message}</span>
                </div>
              )}

              {/* Step-by-step Setup Guide */}
              <div style={{ background: 'var(--bg-glass)', borderRadius: 10, padding: '12px 14px' }}>
                <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: 8 }}>
                  📱 How to receive alerts on your phone:
                </div>
                <ol style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', paddingLeft: 18, lineHeight: 1.8 }}>
                  <li>
                    Install the <strong>ntfy</strong> app from{' '}
                    <a href="https://play.google.com/store/apps/details?id=io.heckel.ntfy" target="_blank" rel="noreferrer" style={{ color: 'var(--accent-cyan)' }}>Google Play</a>{' '}
                    or <a href="https://apps.apple.com/app/ntfy/id1625396347" target="_blank" rel="noreferrer" style={{ color: 'var(--accent-cyan)' }}>Apple App Store</a>.
                  </li>
                  <li>
                    In the app, tap <strong>"+"</strong> / <strong>Subscribe to topic</strong>.
                  </li>
                  <li>
                    Enter topic name: <code style={{ fontFamily: 'var(--font-mono)', color: '#38bdf8', fontWeight: 700 }}>{ntfyTopic}</code> and subscribe.
                  </li>
                  <li>
                    Done! Incidents in live feed with &ge; 70% decision score (video+audio) or &ge; 80% (single modality) will alert you instantly with sound and images.
                  </li>
                </ol>
              </div>
            </div>
          </div>

          {/* Owner Alert Toggles */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Subscriber Alert Toggles</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                {owners.filter(o => o.active).length}/{owners.length} active
              </span>
            </div>
            {owners.length === 0 ? (
              <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                No subscribers registered. <a href="/owners">Add one →</a>
              </p>
            ) : (
              owners.map(o => (
                <div className="toggle-row" key={o.id}>
                  <div className="toggle-info">
                    <h4>
                      {channelEmoji(o.channel)} {o.name}
                    </h4>
                    <p>{o.phone} · threshold {Math.round(o.alert_threshold * 100)}% · cooldown {o.alert_cooldown_sec}s</p>
                  </div>
                  <label className="toggle">
                    <input type="checkbox" checked={o.active} onChange={() => toggleOwner(o)} />
                    <span className="toggle-slider" />
                  </label>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Right: System Stats & Endpoints */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div className="card">
            <div className="card-header">
              <span className="card-title">Alert Statistics</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              {[
                { icon: '🚨', label: 'Total Incidents', value: totalIncidents },
                { icon: '📩', label: 'Alerts Sent', value: totalAlerts },
                { icon: '🔔', label: 'Active Subscribers', value: owners.filter(o => o.active).length },
                { icon: '📈', label: 'Alert / Incident', value: successRate === '—' ? '—' : `${successRate}%` },
              ].map(({ icon, label, value }) => (
                <div key={label} style={{ background: 'var(--bg-glass)', borderRadius: 10, padding: '14px 16px', textAlign: 'center' }}>
                  <div style={{ fontSize: '1.6rem', marginBottom: 4 }}>{icon}</div>
                  <div style={{ fontSize: '1.4rem', fontWeight: 800 }}>{value}</div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>{label}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">API Endpoints Reference</span>
            </div>
            {[
              ['POST', '/api/notifications/test-ntfy', 'Trigger ntfy test alert (sample.gif)'],
              ['GET',  '/api/notifications/settings',  'Get ntfy config & URLs'],
              ['POST', '/api/notifications/send',      'Send immediate ntfy push'],
              ['POST', '/api/camera/start',            'Start analysis'],
              ['POST', '/api/camera/stop',             'Stop analysis'],
              ['GET',  '/api/camera/status',           'Camera status'],
              ['GET',  '/api/camera/stream',           'MJPEG stream'],
              ['GET',  '/api/owners',                  'List subscribers'],
              ['POST', '/api/owners',                  'Register subscriber'],
              ['GET',  '/api/incidents',               'Incident history'],
              ['WS',   'ws://localhost:8000/ws/incidents', 'Real-time WebSocket feed'],
            ].map(([method, path, desc]) => (
              <div key={path} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '7px 0', borderBottom: '1px solid var(--border)' }}>
                <span style={{
                  fontFamily: 'var(--font-mono)', fontSize: '0.68rem', fontWeight: 700,
                  padding: '2px 7px', borderRadius: 4,
                  background: method === 'GET' ? 'rgba(16,185,129,0.15)' : method === 'POST' ? 'rgba(59,130,246,0.15)' : 'rgba(168,85,247,0.15)',
                  color: method === 'GET' ? 'var(--accent-green)' : method === 'POST' ? 'var(--accent-blue)' : 'var(--accent-purple)',
                  flexShrink: 0,
                }}>
                  {method}
                </span>
                <code style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', flex: 1 }}>{path}</code>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{desc}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
