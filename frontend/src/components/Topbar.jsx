import { Bell, ChevronRight, Clock3, ShieldCheck } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useLocation } from 'react-router-dom'

const TITLES = {
  '/': 'Dashboard',
  '/live': 'Live Feed',
  '/authorized': 'Authorized Users',
  '/authorized-users': 'Authorized Users',
  '/owners': 'Owners',
  '/incidents': 'Incident History',
  '/settings': 'Alert Settings',
}

export default function Topbar() {
  const { pathname } = useLocation()
  const [time, setTime] = useState(new Date())

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  return (
    <header className="topbar">
      <div className="breadcrumb">
        <ShieldCheck size={19} />
        <span>Security operations</span>
        <ChevronRight size={17} />
        <strong>{TITLES[pathname] || 'Dashboard'}</strong>
      </div>
      <div className="topbar-actions">
        <span className="utc-clock"><Clock3 size={18} /> {time.toLocaleTimeString('en-GB', { hour12: false })} <em>UTC</em></span>
        <button className="topbar-icon" aria-label="Notifications"><Bell size={19} /><i /></button>
        <span className="profile-avatar">AL</span>
      </div>
    </header>
  )
}
