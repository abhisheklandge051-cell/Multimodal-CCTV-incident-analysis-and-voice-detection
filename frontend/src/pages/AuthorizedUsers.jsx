// src/pages/AuthorizedUsers.jsx
import { useEffect, useState, useCallback, useRef } from 'react'
import toast from 'react-hot-toast'
import { useWS } from '../context/WebSocketContext'
import { formatDateTime } from '../utils/incidentMeta'

const API = '/api/auth-users'
const CAM_STATUS_API = '/api/camera/status'

function EnrollModal({ onClose, onSaved }) {
  const [name, setName] = useState('')
  const [role, setRole] = useState('Authorized Personnel')
  const [photoB64, setPhotoB64] = useState('')
  const [photoPreview, setPhotoPreview] = useState('')
  const [saving, setSaving] = useState(false)
  const [capturingCam, setCapturingCam] = useState(false)
  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (evt) => {
      const dataUrl = evt.target?.result
      setPhotoB64(dataUrl)
      setPhotoPreview(dataUrl)
    }
    reader.readAsDataURL(file)
  }

  const handleCaptureLive = async () => {
    setCapturingCam(true)
    try {
      const res = await fetch(`${API}/snapshot-live`)
      if (!res.ok) {
        const err = await res.json()
        toast.error(err.detail || 'Failed to capture from camera stream', { className: 'custom-toast' })
      } else {
        const data = await res.json()
        setPhotoB64(data.photo_b64)
        setPhotoPreview(data.photo_b64)
        toast.success('Captured snapshot from live camera feed!', { className: 'custom-toast' })
      }
    } catch {
      toast.error('Could not connect to camera stream', { className: 'custom-toast' })
    }
    setCapturingCam(false)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!photoB64) {
      toast.error('Please upload a face photo or capture from live camera', { className: 'custom-toast' })
      return
    }

    setSaving(true)
    try {
      const res = await fetch(API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name.trim(),
          role: role.trim() || 'Authorized Personnel',
          photo_b64: photoB64,
          active: true,
        }),
      })

      if (!res.ok) {
        const err = await res.json()
        toast.error(err.detail || 'Enrollment failed. Ensure face is clearly visible.', { className: 'custom-toast' })
      } else {
        toast.success(`Authorized user '${name}' enrolled successfully!`, { className: 'custom-toast', icon: '🛡️' })
        onSaved()
      }
    } catch {
      toast.error('Network error during enrollment', { className: 'custom-toast' })
    }
    setSaving(false)
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" style={{ maxWidth: 520 }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>🛡️ Enroll Authorized Personnel</h2>
          <button className="btn btn-ghost btn-icon btn-sm" onClick={onClose}>✕</button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label">Full Name / ID</label>
            <input
              required
              className="form-input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Officer John / Abhishek"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Designation / Role</label>
            <input
              className="form-input"
              value={role}
              onChange={(e) => setRole(e.target.value)}
              placeholder="e.g. Site Security, Admin, Maintenance"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Face Biometric Enrollment</label>
            <div style={{ display: 'flex', gap: 10, marginBottom: 12 }}>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => fileInputRef.current?.click()}
                style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}
              >
                📁 Upload Photo
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                style={{ display: 'none' }}
                onChange={handleFileChange}
              />

              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={handleCaptureLive}
                disabled={capturingCam}
                style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}
              >
                📸 {capturingCam ? 'Capturing…' : 'Capture Live Feed'}
              </button>
            </div>

            {/* Photo Preview Card */}
            {photoPreview ? (
              <div style={{
                position: 'relative',
                background: 'rgba(0,0,0,0.4)',
                border: '1px solid rgba(16,185,129,0.4)',
                borderRadius: 10,
                padding: 10,
                display: 'flex',
                alignItems: 'center',
                gap: 14
              }}>
                <img
                  src={photoPreview}
                  alt="Enrolled Face"
                  style={{ width: 80, height: 80, objectFit: 'cover', borderRadius: 8, border: '2px solid #10b981' }}
                />
                <div style={{ flex: 1 }}>
                  <div style={{ color: '#10b981', fontWeight: 600, fontSize: '0.85rem' }}>
                    ✓ Face Photo Ready
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: 2 }}>
                    Face embeddings will be computed and cached for high-speed live bypass.
                  </div>
                </div>
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  onClick={() => { setPhotoB64(''); setPhotoPreview(''); }}
                  style={{ color: '#f87171' }}
                >
                  Remove
                </button>
              </div>
            ) : (
              <div style={{
                border: '2px dashed var(--border)',
                borderRadius: 10,
                padding: '24px 16px',
                textAlign: 'center',
                background: 'var(--bg-glass)',
                color: 'var(--text-secondary)',
                fontSize: '0.82rem',
              }}>
                <span style={{ fontSize: '1.6rem', display: 'block', marginBottom: 6 }}>👤</span>
                Upload a clear frontal face image or 1-click snapshot from the active camera.
              </div>
            )}
          </div>

          <div style={{
            background: 'rgba(6,182,212,0.08)',
            border: '1px solid rgba(6,182,212,0.2)',
            borderRadius: 8,
            padding: '10px 12px',
            fontSize: '0.76rem',
            color: 'var(--text-secondary)',
            marginBottom: 18,
            lineHeight: 1.4
          }}>
            ℹ️ <strong style={{ color: '#38bdf8' }}>Incident Bypass Rule:</strong> When this authorized person is detected in the CCTV feed, downstream incident detection and alerts (fighting, car accident, etc.) are automatically suppressed to avoid false alarms.
          </div>

          <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
            <button type="button" className="btn btn-ghost" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={saving || !photoB64}>
              {saving ? 'Verifying & Enrolling…' : 'Enroll Personnel'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function AuthorizedUsers() {
  const { connected } = useWS()
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(true)
  const [showModal, setShowModal] = useState(false)
  const [camStatus, setCamStatus] = useState({ running: false, auth_bypass: false, authorized_users: [] })

  const fetchUsers = useCallback(async () => {
    try {
      const res = await fetch(API)
      if (res.ok) {
        const data = await res.json()
        setUsers(data)
      }
    } catch {
      toast.error('Failed to load authorized users', { className: 'custom-toast' })
    }
    setLoading(false)
  }, [])

  const fetchCamStatus = useCallback(async () => {
    try {
      const res = await fetch(CAM_STATUS_API)
      if (res.ok) {
        const data = await res.json()
        setCamStatus(data)
      }
    } catch {}
  }, [])

  useEffect(() => {
    fetchUsers()
    fetchCamStatus()
    const iv = setInterval(fetchCamStatus, 3000)
    return () => clearInterval(iv)
  }, [fetchUsers, fetchCamStatus])

  const handleToggle = async (user) => {
    try {
      const res = await fetch(`${API}/${user.id}/toggle`, { method: 'PATCH' })
      if (res.ok) {
        toast.success(`User '${user.name}' is now ${!user.active ? 'active' : 'inactive'}`, { className: 'custom-toast' })
        fetchUsers()
      }
    } catch {
      toast.error('Failed to toggle user status', { className: 'custom-toast' })
    }
  }

  const handleDelete = async (user) => {
    if (!window.confirm(`Are you sure you want to remove authorized personnel '${user.name}'?`)) return
    try {
      const res = await fetch(`${API}/${user.id}`, { method: 'DELETE' })
      if (res.ok || res.status === 204) {
        toast.success(`Removed authorized personnel '${user.name}'`, { className: 'custom-toast' })
        fetchUsers()
      }
    } catch {
      toast.error('Failed to delete user', { className: 'custom-toast' })
    }
  }

  const activeCount = users.filter((u) => u.active).length
  const liveAuthorizedUsers = camStatus.authorized_users || []

  return (
    <div>
      <div className="page-header">
        <div className="page-title-group">
          <h1>Authorized Personnel</h1>
          <p>Personnel enrolled to automatically bypass incident detection and suppress false alarms in live feed</p>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <button className="btn btn-primary" onClick={() => setShowModal(true)}>
            ➕ Enroll Personnel
          </button>
        </div>
      </div>

      {/* Status Highlights */}
      <div className="stats-grid" style={{ marginBottom: 24 }}>
        <div className="stat-card">
          <div className="stat-label">Total Enrolled</div>
          <div className="stat-val" style={{ color: 'var(--accent-blue)' }}>{users.length}</div>
          <div className="stat-sub">Trusted personnel</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Active Whitelist</div>
          <div className="stat-val" style={{ color: '#10b981' }}>{activeCount}</div>
          <div className="stat-sub">Currently bypassing detection</div>
        </div>

        <div className="stat-card" style={{
          border: camStatus.auth_bypass ? '1px solid rgba(16,185,129,0.5)' : '1px solid var(--border)',
          background: camStatus.auth_bypass ? 'rgba(16,185,129,0.08)' : 'var(--bg-card)'
        }}>
          <div className="stat-label">Live Camera Bypass Status</div>
          <div className="stat-val" style={{
            fontSize: '1.25rem',
            color: camStatus.auth_bypass ? '#10b981' : 'var(--text-muted)'
          }}>
            {camStatus.auth_bypass
              ? `🟢 BYPASS ACTIVE: ${liveAuthorizedUsers.join(', ')}`
              : '⚪ MONITORING (No Authorized User in View)'}
          </div>
          <div className="stat-sub">
            {camStatus.running ? '● Camera active' : 'Camera off'} · {connected ? 'WS synced' : 'Offline'}
          </div>
        </div>
      </div>

      {/* Personnel Table */}
      <div className="card">
        <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span style={{ fontWeight: 700, fontSize: '1rem', color: 'var(--text-primary)' }}>
              Enrolled Personnel List
            </span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginLeft: 10 }}>
              ({users.length} registered)
            </span>
          </div>
        </div>

        {loading ? (
          <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
            Loading authorized personnel…
          </div>
        ) : users.length === 0 ? (
          <div style={{ padding: 48, textAlign: 'center', color: 'var(--text-muted)' }}>
            <span style={{ fontSize: '2.4rem', display: 'block', marginBottom: 12 }}>🛡️</span>
            <strong>No authorized personnel enrolled yet.</strong>
            <p style={{ marginTop: 6, fontSize: '0.85rem' }}>
              Add authorized users to allow them to bypass incident detection when present in front of the CCTV camera.
            </p>
            <button className="btn btn-primary" style={{ marginTop: 16 }} onClick={() => setShowModal(true)}>
              ➕ Enroll First Personnel
            </button>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="table">
              <thead>
                <tr>
                  <th style={{ width: 64 }}>Photo</th>
                  <th>Name</th>
                  <th>Designation / Role</th>
                  <th>Status</th>
                  <th>Enrolled On</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.map((user) => {
                  const isVisibleNow = liveAuthorizedUsers.some(
                    (name) => name.toLowerCase() === user.name.toLowerCase()
                  )

                  return (
                    <tr key={user.id}>
                      <td>
                        {user.photo_b64 ? (
                          <img
                            src={user.photo_b64}
                            alt={user.name}
                            style={{
                              width: 44,
                              height: 44,
                              objectFit: 'cover',
                              borderRadius: '50%',
                              border: isVisibleNow ? '2px solid #10b981' : '1px solid var(--border)',
                              boxShadow: isVisibleNow ? '0 0 10px rgba(16,185,129,0.5)' : 'none',
                            }}
                          />
                        ) : (
                          <div style={{
                            width: 44,
                            height: 44,
                            borderRadius: '50%',
                            background: 'var(--bg-glass)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '1.2rem',
                          }}>
                            👤
                          </div>
                        )}
                      </td>
                      <td>
                        <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                          {user.name}
                        </div>
                        {isVisibleNow && (
                          <span style={{
                            fontSize: '0.68rem',
                            fontWeight: 700,
                            color: '#10b981',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 4,
                            marginTop: 2
                          }}>
                            ● LIVE IN FEED (BYPASSING)
                          </span>
                        )}
                      </td>
                      <td>
                        <span className="badge" style={{ background: 'rgba(99,102,241,0.15)', color: '#818cf8', border: '1px solid rgba(99,102,241,0.3)' }}>
                          {user.role || 'Authorized Personnel'}
                        </span>
                      </td>
                      <td>
                        <button
                          type="button"
                          onClick={() => handleToggle(user)}
                          style={{
                            cursor: 'pointer',
                            background: 'none',
                            border: 'none',
                            padding: 0,
                          }}
                        >
                          {user.active ? (
                            <span className="badge badge-active">● Active (Bypassing)</span>
                          ) : (
                            <span className="badge badge-inactive">○ Inactive (Monitored)</span>
                          )}
                        </button>
                      </td>
                      <td style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        {formatDateTime(user.created_at)}
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        <button
                          className="btn btn-ghost btn-sm"
                          style={{ color: '#f87171' }}
                          onClick={() => handleDelete(user)}
                          title="Remove user"
                        >
                          🗑️ Delete
                        </button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {showModal && (
        <EnrollModal
          onClose={() => setShowModal(false)}
          onSaved={() => {
            setShowModal(false)
            fetchUsers()
          }}
        />
      )}
    </div>
  )
}
