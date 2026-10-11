import { useEffect, useState, useRef, useCallback } from 'react'
import { Activity, ArrowUpRight, Camera, CarFront, CheckCircle2, CircleAlert, Flame, Layers3, Mic, MonitorPlay, Radio, ScanSearch, Settings2, ShieldAlert, TriangleAlert, Video, VideoOff } from 'lucide-react'
import { useWS } from '../context/WebSocketContext'
import { getIncidentMeta, getSeverityClass, formatTime } from '../utils/incidentMeta'

const API = '/api'
const STREAM_URL = '/api/camera/stream'

const MODES = [
  { id: 'multimodal', label: 'Multimodal', Icon: Layers3, desc: 'Video + Audio fusion', config: { enable_yolo: true, enable_video_incident: true, enable_audio_incident: true } },
  { id: 'video', label: 'Video only', Icon: Video, desc: 'GRU-ResNet + YOLO', config: { enable_yolo: true, enable_video_incident: true, enable_audio_incident: false } },
  { id: 'audio', label: 'Audio only', Icon: Mic, desc: 'Keras audio model', config: { enable_yolo: false, enable_video_incident: false, enable_audio_incident: true } },
  { id: 'custom', label: 'Custom', Icon: Settings2, desc: 'Configure manually', config: null },
]

const INCIDENT_ICONS = { Abuse: ShieldAlert, CarAccident: CarFront, Explosion: TriangleAlert, Fighting: CircleAlert, Normal: CheckCircle2, Riot: Flame, Shooting: ShieldAlert }

function StatCard({ Icon, tone, value, unit, label, note, status }) {
  return <article className="overview-stat-card"><div className={`overview-icon ${tone}`}><Icon size={22} /></div><p>{label}</p><div className="overview-value">{value}<small>{unit}</small></div><div className="overview-note"><span>{note}</span><strong className={status.className}>{status.text}</strong></div></article>
}

function IncidentRow({ ev }) {
  const meta = getIncidentMeta(ev.label)
  const Icon = INCIDENT_ICONS[ev.label] || CircleAlert
  return <article className={`recent-incident ${getSeverityClass(ev.label)}`}><div className="recent-incident-icon" style={{ color: meta.color, background: `${meta.color}16` }}><Icon size={19} /></div><div className="recent-incident-body"><strong>{ev.label} detected</strong><span>{ev.modality_status || 'Camera analysis'} · {formatTime(ev.timestamp)}</span></div><div className="recent-incident-confidence" style={{ color: meta.color }}>{(ev.confidence * 100).toFixed(0)}%<small>confidence</small></div></article>
}

function ModePills({ selected, onChange }) {
  return <div className="analysis-mode-pills">{MODES.map(({ id, label, Icon, desc }) => <button key={id} title={desc} onClick={() => onChange(id)} className={selected === id ? 'selected' : ''}><Icon size={15} />{label}</button>)}</div>
}

export default function SecurityDashboard() {
  const { incidents, connected } = useWS()
  const [stats, setStats] = useState(null)
  const [camStatus, setCamStatus] = useState({ running: false })
  const [loading, setLoading] = useState(false)
  const [starting, setStarting] = useState(false)
  const [pipelineError, setPipelineError] = useState(null)
  const [modeId, setModeId] = useState('multimodal')
  const [source, setSource] = useState('0')
  const [customForm, setCustomForm] = useState({ enable_yolo: true, enable_video_incident: true, enable_audio_incident: true, video_weight: 0.55, audio_weight: 0.45, yolo_conf: 0.35 })
  const startPollRef = useRef(null)

  const fetchStatus = useCallback(() => fetch(`${API}/camera/status`).then(r => r.json()).then(d => { setCamStatus(d); setPipelineError(d.pipeline_error || null) }).catch(() => {}), [])
  useEffect(() => { fetchStatus(); fetch(`${API}/incidents/stats`).then(r => r.json()).then(setStats).catch(() => {}); const interval = setInterval(() => { fetchStatus(); fetch(`${API}/incidents/stats`).then(r => r.json()).then(setStats).catch(() => {}) }, 3000); return () => clearInterval(interval) }, [fetchStatus])
  useEffect(() => () => clearTimeout(startPollRef.current), [])

  const startStartupPoll = useCallback(() => {
    let attempts = 0
    const poll = () => { attempts += 1; fetch(`${API}/camera/status`).then(r => r.json()).then(d => { setCamStatus(d); if (d.pipeline_error) { setPipelineError(d.pipeline_error); setStarting(false) } else if (d.running) setStarting(false); else if (attempts < 60) startPollRef.current = setTimeout(poll, 1000); else setStarting(false) }).catch(() => { if (attempts < 60) startPollRef.current = setTimeout(poll, 1000); else setStarting(false) }) }
    startPollRef.current = setTimeout(poll, 1000)
  }, [])

  const buildPayload = () => {
    if (modeId === 'custom') return { ...customForm, source }
    const selected = MODES.find(m => m.id === modeId)
    const weights = modeId === 'audio' ? { video_weight: 0, audio_weight: 1 } : modeId === 'video' ? { video_weight: 1, audio_weight: 0 } : { video_weight: 0.55, audio_weight: 0.45 }
    return { source, ...selected.config, ...weights, yolo_conf: 0.35 }
  }

  const handleStart = async () => { setLoading(true); setPipelineError(null); try { const res = await fetch(`${API}/camera/start`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(buildPayload()) }); if (!res.ok) { const err = await res.json(); setPipelineError(err.detail || 'Failed to start analysis'); return } setCamStatus(await res.json()); setStarting(true); startStartupPoll() } catch (error) { setPipelineError(`Network error: ${error.message}`) } finally { setLoading(false) } }
  const handleStop = async () => { setLoading(true); clearTimeout(startPollRef.current); setStarting(false); try { const res = await fetch(`${API}/camera/stop`, { method: 'POST' }); setCamStatus(await res.json()) } catch (_) {} finally { setLoading(false) } }

  const activeIncidents = incidents.filter(event => event.label !== 'Normal')
  const incidentClasses = ['Abuse', 'CarAccident', 'Explosion', 'Fighting', 'Riot', 'Shooting']
  const todayTotal = Object.values(stats?.today_by_class || {}).reduce((sum, value) => sum + value, 0)

  return <div className="security-dashboard">
    <section className="security-overview-heading"><div><span className="eyebrow"><i /> Security operations center</span><h1>Security overview</h1><p>A clear view of your perimeter. Every camera, every incident.</p></div><div className="overview-actions"><span className={`connection-chip ${connected ? 'online' : ''}`}><i /> {connected ? 'System online' : 'Reconnecting'}</span><a className="btn btn-primary" href="/live"><Radio size={17} />Live feed <ArrowUpRight size={16} /></a></div></section>
    <section className="overview-stat-grid"><StatCard Icon={Camera} tone="blue" value={camStatus.running ? '01' : '00'} unit="/ 01" label="Active cameras" note={camStatus.running ? `${camStatus.fps || 0} FPS live analysis` : 'Camera on standby'} status={{ className: camStatus.running ? 'positive' : '', text: camStatus.running ? 'Live' : 'Standby' }} /><StatCard Icon={TriangleAlert} tone="red" value={String(activeIncidents.length).padStart(2, '0')} label="Open incidents" note={activeIncidents.length ? 'Review required' : 'No active threats'} status={{ className: activeIncidents.length ? 'critical' : 'positive', text: activeIncidents.length ? 'Action required' : 'All clear' }} /><StatCard Icon={ShieldAlert} tone="green" value={connected ? '100' : '—'} unit={connected ? '%' : ''} label="Detection service" note="Multimodal security analysis" status={{ className: connected ? 'positive' : 'critical', text: connected ? 'Protected' : 'Offline' }} /><StatCard Icon={Activity} tone="amber" value={camStatus.running ? Math.floor((camStatus.uptime_sec || 0) / 60) : '—'} unit={camStatus.running ? ' min' : ''} label="System uptime" note={camStatus.running ? 'Current monitoring session' : 'Monitoring not started'} status={{ className: 'positive', text: camStatus.running ? 'Operational' : 'Idle' }} /></section>
    <section className="security-dashboard-grid"><article className="dashboard-camera-card card"><header className="camera-card-header"><div className="camera-title"><span><Camera size={20} /></span><div><h2>North Gate</h2><p><i className={camStatus.running ? 'live-dot' : ''} /> {camStatus.running ? 'Live analysis active' : 'Camera standby'} <b>·</b> {camStatus.running ? `${camStatus.fps || 0} FPS` : 'Ready to monitor'}</p></div></div><span className={`camera-state ${camStatus.running ? 'live' : ''}`}>{camStatus.running ? 'LIVE' : 'STANDBY'}</span></header><div className="security-camera-preview">{camStatus.running ? <img src={STREAM_URL} alt="Live North Gate security camera" /> : <div className="camera-empty"><VideoOff size={38} /><strong>Camera monitoring is on standby</strong><span>Select an analysis mode and start the secure feed.</span></div>}{starting && <div className="camera-loading"><span /><strong>Initialising security models</strong></div>}</div>{pipelineError && <div className="pipeline-error"><TriangleAlert size={17} />{pipelineError}</div>}<div className="camera-control-panel">{!camStatus.running && !starting && <><div className="camera-control-top"><label>Camera source<input className="form-input" value={source} onChange={event => setSource(event.target.value)} placeholder="0 or /path/to/video.mp4" /></label><div><label>Analysis mode</label><ModePills selected={modeId} onChange={setModeId} /></div></div>{modeId === 'custom' && <details className="advanced-controls"><summary><Settings2 size={16} />Advanced analysis controls</summary><div className="advanced-grid"><label>Video weight <input type="range" min="0" max="1" step="0.05" value={customForm.video_weight} onChange={event => setCustomForm(value => ({ ...value, video_weight: +event.target.value }))} /></label><label>Audio weight <input type="range" min="0" max="1" step="0.05" value={customForm.audio_weight} onChange={event => setCustomForm(value => ({ ...value, audio_weight: +event.target.value }))} /></label><label>YOLO threshold <input type="range" min="0.1" max="0.95" step="0.05" value={customForm.yolo_conf} onChange={event => setCustomForm(value => ({ ...value, yolo_conf: +event.target.value }))} /></label></div></details>}<button className="btn btn-primary camera-action" onClick={handleStart} disabled={loading}><MonitorPlay size={18} />{loading ? 'Sending request' : 'Start secure analysis'}</button></>}{(camStatus.running || starting) && <div className="camera-running-bar"><span><Radio size={17} />{starting ? 'Loading detection models' : `${camStatus.frame_count?.toLocaleString() || 0} frames analysed`}</span><button className="btn btn-danger" onClick={handleStop} disabled={loading}>{loading ? 'Stopping' : 'Stop analysis'}</button></div>}</div></article><aside className="recent-incidents-panel"><header><div><h2>Recent incidents</h2><span>{activeIncidents.length}</span></div><a href="/incidents">View all <ArrowUpRight size={15} /></a></header><div className="recent-incidents-list">{activeIncidents.length ? activeIncidents.slice(0, 4).map((event, index) => <IncidentRow key={`${event.incident_id || event.timestamp}-${index}`} ev={event} />) : <div className="dashboard-empty"><CheckCircle2 size={29} /><strong>Perimeter clear</strong><span>No incidents have been detected.</span></div>}</div></aside></section>
    <section className="dashboard-lower-grid"><article className="card incident-distribution"><header><div><span className="card-title">Today's activity</span><h2>Incident distribution</h2></div><ScanSearch size={20} /></header>{incidentClasses.map(label => { const count = stats?.today_by_class?.[label] || 0; const meta = getIncidentMeta(label); const pct = todayTotal ? (count / todayTotal) * 100 : 0; return <div className="distribution-row" key={label}><span>{label}</span><div><i style={{ width: `${pct}%`, background: meta.color }} /></div><b>{count}</b></div> })}</article><article className="card system-status-card"><header><span className="card-title">System status</span><ShieldAlert size={20} /></header><div className="status-line"><span><i className={connected ? 'ok' : 'warn'} />Secure websocket</span><b>{connected ? 'Connected' : 'Checking'}</b></div><div className="status-line"><span><i className={camStatus.running ? 'ok' : 'warn'} />Camera pipeline</span><b>{camStatus.running ? 'Monitoring' : 'Standby'}</b></div><div className="status-line"><span><i className="ok" />Alert service</span><b>Ready</b></div></article></section>
  </div>
}
