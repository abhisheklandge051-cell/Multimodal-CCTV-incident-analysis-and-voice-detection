// src/components/Sidebar.jsx
import { NavLink } from 'react-router-dom'
import { BellRing, History, LayoutDashboard, Radio, ShieldCheck, UsersRound, Wifi } from 'lucide-react'
import { useWS } from '../context/WebSocketContext'

const NAV = [
  { to: '/',            icon: LayoutDashboard, label: 'Dashboard'        },
  { to: '/live',        icon: Radio,           label: 'Live Feed'        },
  { to: '/authorized',  icon: ShieldCheck,     label: 'Authorized Users' },
  { to: '/owners',      icon: UsersRound,      label: 'Owners'           },
  { to: '/incidents',   icon: History,          label: 'Incident History' },
  { to: '/settings',    icon: BellRing,         label: 'Alert Settings'   },
]

export default function Sidebar() {
  const { connected } = useWS()

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon"><ShieldCheck size={23} strokeWidth={2.4} /></div>
        <div className="sidebar-logo-text">
          <strong>SecureShield</strong>
          <span>CCTV INCIDENT MONITOR</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-section-label">Workspace</div>
        {NAV.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
          >
            <Icon className="nav-icon" size={19} strokeWidth={1.8} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="protection-status">
          <ShieldCheck size={21} />
          <div><strong>Perimeter protected</strong><span>All systems operational</span></div>
          <i />
        </div>
        <div className="ws-status">
          <Wifi size={14} />
          <span>{connected ? 'WebSocket live' : 'Reconnecting…'}</span>
        </div>
      </div>
    </aside>
  )
}
