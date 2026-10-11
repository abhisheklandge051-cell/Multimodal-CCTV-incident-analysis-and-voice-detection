// src/pages/OwnersPage.jsx
import { useEffect, useState } from 'react'
import toast from 'react-hot-toast'
import { formatDateTime } from '../utils/incidentMeta'

const API = '/api/owners'

const EMPTY_FORM = {
  name: '', phone: 'incident_alert_shubh', channel: 'ntfy', active: true,
  alert_threshold: 0.80, alert_cooldown_sec: 60
}

function OwnerModal({ owner, onClose, onSaved }) {
  const [form, setForm] = useState(owner ? { ...owner } : { ...EMPTY_FORM })
  const [saving, setSaving] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setSaving(true)
    try {
      const method = owner ? 'PATCH' : 'POST'
      const url = owner ? `${API}/${owner.id}` : API
      const res = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      if (!res.ok) {
        const err = await res.json()
        toast.error(err.detail || 'Failed to save owner', { className: 'custom-toast' })
      } else {
        toast.success(owner ? 'Owner updated!' : 'Owner registered!', { className: 'custom-toast' })
        onSaved()
      }
    } catch (err) {
      toast.error('Network error', { className: 'custom-toast' })
    }
    setSaving(false)
  }

  const set = (k, v) => setForm(p => ({ ...p, [k]: v }))

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>{owner ? '✏️ Edit Owner' : '➕ Register Owner'}</h2>
          <button className="btn btn-ghost btn-icon btn-sm" onClick={onClose}>✕</button>
        </div>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label">Full Name</label>
            <input required className="form-input" value={form.name}
              onChange={e => set('name', e.target.value)} placeholder="Security Desk / Shubh" />
          </div>
          <div className="form-group">
            <label className="form-label">{form.channel === 'ntfy' ? 'ntfy.sh Topic' : 'Phone (E.164 format)'}</label>
            <input required className="form-input" value={form.phone}
              onChange={e => set('phone', e.target.value)}
              placeholder={form.channel === 'ntfy' ? 'incident_alert_shubh' : '+919876543210'} />
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              {form.channel === 'ntfy'
                ? 'ntfy.sh topic subscribed to on mobile or web application'
                : 'Include country code: +91 for India, +1 for USA, etc.'}
            </span>
          </div>
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Alert Channel</label>
              <select className="form-select" value={form.channel} onChange={e => set('channel', e.target.value)}>
                <option value="ntfy">🔔 ntfy.sh Topic</option>
                <option value="sms">📱 SMS</option>
                <option value="whatsapp">💬 WhatsApp</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Status</label>
              <select className="form-select" value={form.active ? 'active' : 'inactive'} onChange={e => set('active', e.target.value === 'active')}>
                <option value="active">✅ Active</option>
                <option value="inactive">⏸ Inactive</option>
              </select>
            </div>
          </div>
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Alert Threshold ({Math.round(form.alert_threshold * 100)}%)</label>
              <input type="range" min="0" max="1" step="0.05" value={form.alert_threshold}
                onChange={e => set('alert_threshold', +e.target.value)}
                style={{ width: '100%', accentColor: 'var(--accent-blue)' }} />
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                Only alert if confidence ≥ {Math.round(form.alert_threshold * 100)}%
              </span>
            </div>
            <div className="form-group">
              <label className="form-label">Cooldown (sec)</label>
              <input type="number" className="form-input" min="5" max="3600" value={form.alert_cooldown_sec}
                onChange={e => set('alert_cooldown_sec', +e.target.value)} />
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                Min wait between same-class alerts
              </span>
            </div>
          </div>
          <div className="modal-actions">
            <button type="button" className="btn btn-ghost" onClick={onClose}>Cancel</button>
            <button type="submit" className="btn btn-primary" disabled={saving}>
              {saving ? 'Saving…' : (owner ? 'Save Changes' : 'Register Owner')}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function OwnersPage() {
  const [owners, setOwners] = useState([])
  const [loading, setLoading] = useState(true)
  const [modal, setModal] = useState(null) // null | 'new' | owner object

  const fetchOwners = () => {
    setLoading(true)
    fetch(API).then(r => r.json()).then(setOwners).catch(() => {}).finally(() => setLoading(false))
  }

  useEffect(() => { fetchOwners() }, [])

  const handleDelete = async (owner) => {
    if (!confirm(`Delete owner "${owner.name}"? This cannot be undone.`)) return
    try {
      await fetch(`${API}/${owner.id}`, { method: 'DELETE' })
      toast.success('Owner deleted.', { className: 'custom-toast' })
      fetchOwners()
    } catch {
      toast.error('Delete failed.', { className: 'custom-toast' })
    }
  }

  const channelEmoji = (ch) => ch === 'ntfy' ? '🔔' : ch === 'whatsapp' ? '💬' : '📱'

  return (
    <div>
      <div className="page-header">
        <div className="page-title-group">
          <h1>Subscribers &amp; Owners</h1>
          <p>Register and manage contacts who receive incident alerts via ntfy.sh, SMS, or WhatsApp</p>
        </div>
        <button className="btn btn-primary" onClick={() => setModal('new')}>
          + Register Owner
        </button>
      </div>

      {loading ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {[1,2,3].map(i => <div key={i} className="skeleton" style={{ height: 52 }} />)}
        </div>
      ) : owners.length === 0 ? (
        <div className="card">
          <div className="empty-state">
            <span style={{ fontSize: '3rem' }}>👤</span>
            <h3>No owners registered</h3>
            <p>Register an owner to start receiving SMS or WhatsApp incident alerts.</p>
            <button className="btn btn-primary" onClick={() => setModal('new')}>+ Register First Owner</button>
          </div>
        </div>
      ) : (
        <div className="card" style={{ padding: 0 }}>
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Phone</th>
                  <th>Channel</th>
                  <th>Threshold</th>
                  <th>Cooldown</th>
                  <th>Status</th>
                  <th>Registered</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {owners.map(o => (
                  <tr key={o.id}>
                    <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{o.name}</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{o.phone}</td>
                    <td>
                      <span className={`badge badge-${o.channel}`}>
                        {channelEmoji(o.channel)} {o.channel.toUpperCase()}
                      </span>
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{Math.round(o.alert_threshold * 100)}%</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{o.alert_cooldown_sec}s</td>
                    <td>
                      <span className={`badge badge-${o.active ? 'active' : 'inactive'}`}>
                        {o.active ? '✓ Active' : '✗ Inactive'}
                      </span>
                    </td>
                    <td style={{ fontSize: '0.78rem' }}>
                      {formatDateTime(o.created_at)}
                    </td>
                    <td>
                      <div style={{ display: 'flex', gap: 6 }}>
                        <button className="btn btn-ghost btn-sm" onClick={() => setModal(o)}>Edit</button>
                        <button className="btn btn-danger btn-sm" onClick={() => handleDelete(o)}>Delete</button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {modal && (
        <OwnerModal
          owner={modal === 'new' ? null : modal}
          onClose={() => setModal(null)}
          onSaved={() => { setModal(null); fetchOwners() }}
        />
      )}
    </div>
  )
}
