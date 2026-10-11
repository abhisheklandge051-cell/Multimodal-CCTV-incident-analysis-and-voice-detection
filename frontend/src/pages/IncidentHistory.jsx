// src/pages/IncidentHistory.jsx
import { useEffect, useState, useCallback } from 'react'
import { getIncidentMeta, formatDateTime } from '../utils/incidentMeta'

const CLASSES = ['All', 'Abuse', 'CarAccident', 'Explosion', 'Fighting', 'Riot', 'Shooting']
const API = '/api/incidents'

function AlertLogPanel({ incidentId, onClose }) {
  const [detail, setDetail] = useState(null)

  useEffect(() => {
    fetch(`${API}/${incidentId}`).then(r => r.json()).then(setDetail).catch(() => {})
  }, [incidentId])

  const meta = detail ? getIncidentMeta(detail.label) : {}

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" style={{ maxWidth: 620 }} onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Incident #{incidentId} Detail</h2>
          <button className="btn btn-ghost btn-icon btn-sm" onClick={onClose}>✕</button>
        </div>
        {!detail ? (
          <div className="skeleton" style={{ height: 200 }} />
        ) : (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 20 }}>
              {[
                ['Label', <span style={{ color: meta.color, fontWeight: 700 }}>{detail.label}</span>],
                ['Confidence', `${(detail.confidence * 100).toFixed(1)}%`],
                ['Modality', detail.modality_status],
                ['Source', detail.source || '—'],
                ['Video', `${detail.video_label} (${((detail.video_conf || 0)*100).toFixed(0)}%)`],
                ['Audio', `${detail.audio_label} (${((detail.audio_conf || 0)*100).toFixed(0)}%)`],
                ['Time', formatDateTime(detail.timestamp)],
                ['Owners Notified', detail.owners_notified],
              ].map(([k, v]) => (
                <div key={k} style={{ background: 'var(--bg-glass)', borderRadius: 8, padding: '8px 12px' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 2, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{k}</div>
                  <div style={{ fontSize: '0.875rem', color: 'var(--text-primary)', fontFamily: typeof v === 'string' ? 'var(--font-mono)' : 'inherit' }}>{v}</div>
                </div>
              ))}
            </div>
            {detail.snapshot_b64 && (
              <img src={`data:image/jpeg;base64,${detail.snapshot_b64}`} alt="snapshot"
                style={{ width: '100%', borderRadius: 8, marginBottom: 16, maxHeight: 200, objectFit: 'cover' }} />
            )}
            <div className="card-title" style={{ marginBottom: 10 }}>Alert Log ({detail.alert_logs?.length})</div>
            {detail.alert_logs?.length === 0 ? (
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>No alerts were dispatched for this incident.</p>
            ) : (
              <div className="table-wrapper">
                <table>
                  <thead>
                    <tr><th>Recipient / Owner</th><th>Channel</th><th>Status</th><th>Delivery Ref</th><th>Time</th></tr>
                  </thead>
                  <tbody>
                    {detail.alert_logs.map(log => (
                      <tr key={log.id}>
                        <td style={{ color: 'var(--text-primary)' }}>{log.owner_name || `#${log.owner_id}`}</td>
                        <td><span className={`badge badge-${log.channel}`}>{log.channel.toUpperCase()}</span></td>
                        <td><span className={`badge badge-${log.status}`}>{log.status}</span></td>
                        <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                          {log.twilio_sid ? log.twilio_sid.slice(0, 20) + '…' : '—'}
                        </td>
                        <td style={{ fontSize: '0.78rem' }}>{formatDateTime(log.sent_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {detail.alert_logs?.some(l => l.error_message) && (
              <details style={{ marginTop: 10 }}>
                <summary style={{ fontSize: '0.8rem', color: 'var(--text-muted)', cursor: 'pointer' }}>Errors</summary>
                {detail.alert_logs.filter(l => l.error_message).map(l => (
                  <div key={l.id} style={{ fontSize: '0.78rem', color: 'var(--sev-critical)', fontFamily: 'var(--font-mono)', marginTop: 4 }}>
                    {l.owner_name}: {l.error_message}
                  </div>
                ))}
              </details>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default function IncidentHistory() {
  const [incidents, setIncidents] = useState([])
  const [loading, setLoading] = useState(true)
  const [filterClass, setFilterClass] = useState('All')
  const [page, setPage] = useState(0)
  const [selected, setSelected] = useState(null)
  const limit = 30

  const fetchIncidents = useCallback(() => {
    setLoading(true)
    const params = new URLSearchParams({ skip: page * limit, limit })
    if (filterClass !== 'All') params.set('label', filterClass)
    fetch(`${API}?${params}`).then(r => r.json()).then(setIncidents).catch(() => {}).finally(() => setLoading(false))
  }, [page, filterClass])

  useEffect(() => { fetchIncidents() }, [fetchIncidents])

  return (
    <div>
      <div className="page-header">
        <div className="page-title-group">
          <h1>Incident History</h1>
          <p>Complete log of all detected incidents with alert delivery records</p>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <select className="form-select" style={{ width: 160 }} value={filterClass} onChange={e => { setFilterClass(e.target.value); setPage(0) }}>
            {CLASSES.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
          <button className="btn btn-ghost" onClick={fetchIncidents}>↺ Refresh</button>
        </div>
      </div>

      {loading ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {[1,2,3,4,5].map(i => <div key={i} className="skeleton" style={{ height: 48 }} />)}
        </div>
      ) : incidents.length === 0 ? (
        <div className="card">
          <div className="empty-state">
            <span style={{ fontSize: '3rem' }}>📋</span>
            <h3>No incidents recorded</h3>
            <p>Start camera analysis to begin recording incidents.</p>
          </div>
        </div>
      ) : (
        <div className="card" style={{ padding: 0 }}>
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Timestamp</th>
                  <th>Label</th>
                  <th>Confidence</th>
                  <th>Video</th>
                  <th>Audio</th>
                  <th>Modality</th>
                  <th>Notified</th>
                  <th>Source</th>
                  <th>Detail</th>
                </tr>
              </thead>
              <tbody>
                {incidents.map(inc => {
                  const meta = getIncidentMeta(inc.label)
                  return (
                    <tr key={inc.id}>
                      <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>#{inc.id}</td>
                      <td style={{ fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>{formatDateTime(inc.timestamp)}</td>
                      <td>
                        <span className={`badge ${meta.badgeClass}`}>
                          {meta.emoji} {inc.label}
                        </span>
                      </td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div className="conf-bar" style={{ width: 60 }}>
                            <div className="conf-fill" style={{ width: `${inc.confidence * 100}%`, background: meta.color }} />
                          </div>
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem' }}>
                            {(inc.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td style={{ fontSize: '0.8rem' }}>{inc.video_label || '—'}</td>
                      <td style={{ fontSize: '0.8rem' }}>{inc.audio_label || '—'}</td>
                      <td>
                        <span style={{ fontSize: '0.72rem', background: 'var(--bg-glass)', padding: '2px 8px', borderRadius: 4 }}>
                          {inc.modality_status}
                        </span>
                      </td>
                      <td>
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: inc.owners_notified > 0 ? 'var(--accent-green)' : 'var(--text-muted)' }}>
                          {inc.owners_notified}
                        </span>
                      </td>
                      <td style={{ fontSize: '0.78rem', fontFamily: 'var(--font-mono)' }}>{inc.source || '—'}</td>
                      <td>
                        <button className="btn btn-ghost btn-sm" onClick={() => setSelected(inc.id)}>
                          View
                        </button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <div style={{ display: 'flex', gap: 10, padding: '12px 16px', borderTop: '1px solid var(--border)' }}>
            <button className="btn btn-ghost btn-sm" disabled={page === 0} onClick={() => setPage(p => p - 1)}>← Prev</button>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', alignSelf: 'center' }}>Page {page + 1}</span>
            <button className="btn btn-ghost btn-sm" disabled={incidents.length < limit} onClick={() => setPage(p => p + 1)}>Next →</button>
          </div>
        </div>
      )}

      {selected && <AlertLogPanel incidentId={selected} onClose={() => setSelected(null)} />}
    </div>
  )
}
