"""
dashboard/app.py — Live monitoring dashboard for the trading bot.

Endpoints
---------
GET  /                  Dashboard HTML
GET  /api/status        Bot status, market hours, today summary
GET  /api/positions     Open positions from RiskManager
GET  /api/trades        Closed trade history (query: from, to, symbol, direction)
GET  /api/daily-pnl     Last 30 days aggregated P&L
GET  /api/equity-curve  Per-trade cumulative equity
GET  /api/config        Current strategy parameters
POST /api/config        Update strategy parameters live
GET  /api/logs          Last 100 log lines (query: level)
POST /api/control       {action: pause | resume | squareoff}
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from loguru import logger


# ── Shared context (populated by main.py via set_context) ─────────────────────

class _Ctx:
    order_manager = None
    risk_manager  = None
    strategy      = None
    trade_logger  = None
    symbol: str   = "NIFTY"
    bot_status: dict = None

    def __init__(self):
        self.bot_status = {"status": "STOPPED"}


ctx = _Ctx()


def set_context(**kwargs: Any) -> None:
    """Called by main.py to inject live references before dashboard starts."""
    for k, v in kwargs.items():
        setattr(ctx, k, v)


# ── App factory ────────────────────────────────────────────────────────────────

def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def index():
        return render_template_string(_HTML)

    @app.route("/api/status")
    def api_status():
        from config import settings
        today = {}
        if ctx.trade_logger:
            try:
                today = ctx.trade_logger.get_today_summary()
            except Exception:
                pass
        return jsonify({
            "bot_status": ctx.bot_status.get("status", "STOPPED"),
            "mode":       "PAPER" if settings.paper_trade else "LIVE",
            "broker":     settings.broker.upper(),
            "strategy":   settings.strategy,
            "symbol":     ctx.symbol,
            "market":     _market_status(),
            "today":      today,
            "timestamp":  datetime.now().isoformat(),
        })

    @app.route("/api/positions")
    def api_positions():
        if ctx.risk_manager is None:
            return jsonify([])
        try:
            return jsonify([_pos_dict(p) for p in ctx.risk_manager.get_open_positions()])
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/trades")
    def api_trades():
        if ctx.trade_logger is None:
            return jsonify([])
        try:
            from datetime import date
            fd = date.fromisoformat(request.args["from"]) if request.args.get("from") else None
            td = date.fromisoformat(request.args["to"])   if request.args.get("to")   else None
            return jsonify(ctx.trade_logger.get_trades(
                from_date=fd,
                to_date=td,
                symbol=request.args.get("symbol") or None,
                direction=request.args.get("direction") or None,
            ))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/daily-pnl")
    def api_daily_pnl():
        if ctx.trade_logger is None:
            return jsonify([])
        try:
            return jsonify(ctx.trade_logger.get_daily_pnl(days=int(request.args.get("days", 30))))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/equity-curve")
    def api_equity_curve():
        if ctx.trade_logger is None:
            return jsonify([])
        try:
            return jsonify(ctx.trade_logger.get_equity_curve())
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/config", methods=["GET", "POST"])
    def api_config():
        from config import settings
        if request.method == "GET":
            return jsonify({
                "strategy_params": _strategy_cfg(),
                "paper_trade":     settings.paper_trade,
                "capital":         settings.trading_capital,
                "risk_per_trade":  settings.risk_per_trade_pct,
                "max_daily_loss":  settings.max_daily_loss_inr,
                "max_trades_day":  settings.max_trades_per_day,
            })

        body    = request.get_json(silent=True) or {}
        changes = []
        _EDITABLE = {
            "volume_surge_multiplier", "rsi_long_min", "rsi_long_max",
            "rsi_short_min", "rsi_short_max", "sl_atr_multiplier", "rr_ratio",
            "ema_period", "rsi_period", "atr_period", "volume_ma_period",
        }
        if ctx.strategy and hasattr(ctx.strategy, "cfg"):
            cfg = ctx.strategy.cfg
            for key, val in body.items():
                if key in _EDITABLE and hasattr(cfg, key):
                    old = getattr(cfg, key)
                    try:
                        new = type(old)(val)
                        setattr(cfg, key, new)
                        changes.append({"param": key, "old": old, "new": new})
                        if ctx.trade_logger:
                            ctx.trade_logger.log_config_change(key, old, new)
                        logger.info("Config live-updated: {} → {}", key, new)
                    except (ValueError, TypeError) as exc:
                        logger.warning("Config update failed {}: {}", key, exc)
        return jsonify({"updated": changes})

    @app.route("/api/logs")
    def api_logs():
        if ctx.trade_logger is None:
            return jsonify([])
        try:
            return jsonify(ctx.trade_logger.get_recent_logs(
                level=request.args.get("level", "ALL"),
                limit=int(request.args.get("limit", 100)),
            ))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/control", methods=["POST"])
    def api_control():
        body   = request.get_json(silent=True) or {}
        action = body.get("action", "")

        if action == "pause":
            ctx.bot_status["status"] = "PAUSED"
            if ctx.trade_logger:
                ctx.trade_logger.log_bot_event("INFO", "Trading paused via dashboard")
            return jsonify({"ok": True, "status": "PAUSED"})

        if action == "resume":
            ctx.bot_status["status"] = "RUNNING"
            if ctx.trade_logger:
                ctx.trade_logger.log_bot_event("INFO", "Trading resumed via dashboard")
            return jsonify({"ok": True, "status": "RUNNING"})

        if action == "squareoff":
            if ctx.order_manager is None:
                return jsonify({"ok": False, "error": "Bot not running"}), 400
            try:
                ctx.order_manager.square_off_all()
                if ctx.trade_logger:
                    ctx.trade_logger.log_bot_event("WARNING", "Manual square-off via dashboard")
                return jsonify({"ok": True, "message": "All positions squared off"})
            except Exception as exc:
                return jsonify({"ok": False, "error": str(exc)}), 500

        return jsonify({"ok": False, "error": f"Unknown action: {action}"}), 400

    return app


# ── Helpers ────────────────────────────────────────────────────────────────────

def _market_status() -> dict:
    try:
        from data.feed import MarketHours
        mh  = MarketHours()
        now = datetime.now()
        is_open = mh.is_market_open(now)
        return {
            "is_open":       is_open,
            "mins_to_open":  round(mh.minutes_to_open(now),  0) if not is_open else None,
            "mins_to_close": round(mh.minutes_to_close(now), 0) if is_open     else None,
        }
    except Exception:
        return {"is_open": False, "mins_to_open": None, "mins_to_close": None}


def _strategy_cfg() -> dict:
    if ctx.strategy and hasattr(ctx.strategy, "cfg"):
        cfg = ctx.strategy.cfg
        return {k: getattr(cfg, k) for k in vars(cfg) if not k.startswith("_")}
    return {}


def _pos_dict(p) -> dict:
    return {
        "symbol":       p.symbol,
        "direction":    p.direction,
        "entry_price":  p.entry_price,
        "quantity":     p.quantity,
        "current_sl":   p.current_sl,
        "target":       p.target_price,
        "entry_time":   p.entry_time.isoformat() if p.entry_time else None,
        "trail_active": p.trail_activated,
        "partial_done": p.partial_booked,
    }


# ── HTML ───────────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trading Bot Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0d1117;--sf:#161b22;--br:#30363d;--tx:#e6edf3;--mu:#8b949e;
      --ac:#388bfd;--gn:#3fb950;--rd:#f85149;--yw:#d29922}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--tx);font-family:'Segoe UI',system-ui,sans-serif;font-size:14px}
/* header */
header{background:var(--sf);border-bottom:1px solid var(--br);padding:0 24px;
       height:56px;display:flex;align-items:center;gap:10px}
.logo{font-weight:700;font-size:15px}.sp{flex:1}
.ri{font-size:12px;color:var(--mu)}
/* badges */
.badge{padding:3px 9px;border-radius:12px;font-size:11px;font-weight:600;white-space:nowrap}
.b-running{background:rgba(63,185,80,.15);color:var(--gn);border:1px solid rgba(63,185,80,.3)}
.b-paused{background:rgba(210,153,34,.15);color:var(--yw);border:1px solid rgba(210,153,34,.3)}
.b-stopped{background:rgba(248,81,73,.15);color:var(--rd);border:1px solid rgba(248,81,73,.3)}
.b-paper{background:rgba(56,139,253,.15);color:var(--ac);border:1px solid rgba(56,139,253,.3)}
.b-live{background:rgba(248,81,73,.15);color:var(--rd);border:1px solid rgba(248,81,73,.3)}
.b-open{background:rgba(63,185,80,.12);color:var(--gn);border:1px solid rgba(63,185,80,.25)}
.b-closed{background:rgba(139,148,158,.1);color:var(--mu);border:1px solid var(--br)}
/* nav */
nav{background:var(--sf);border-bottom:1px solid var(--br);padding:0 24px;display:flex}
.tb{background:none;border:none;border-bottom:2px solid transparent;color:var(--mu);
    padding:12px 16px;cursor:pointer;font-size:14px;transition:.15s}
.tb:hover{color:var(--tx)}.tb.active{color:var(--ac);border-bottom-color:var(--ac)}
/* layout */
.wrap{padding:24px;max-width:1400px;margin:0 auto}
.panel{display:none}.panel.active{display:block}
/* cards */
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:14px;margin-bottom:20px}
.card{background:var(--sf);border:1px solid var(--br);border-radius:8px;padding:16px}
.cl{font-size:11px;color:var(--mu);text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px}
.cv{font-size:24px;font-weight:700}.cs{font-size:12px;color:var(--mu);margin-top:4px}
.pos{color:var(--gn)}.neg{color:var(--rd)}.neu{color:var(--tx)}
/* buttons */
.ctrls{display:flex;gap:8px;margin-bottom:20px;flex-wrap:wrap}
.btn{padding:8px 15px;border-radius:6px;border:1px solid;cursor:pointer;font-size:13px;
     font-weight:500;background:none;transition:.15s}
.btn:hover{opacity:.8}.btn:disabled{opacity:.35;cursor:not-allowed}
.btn-pause{color:var(--yw);border-color:rgba(210,153,34,.4);background:rgba(210,153,34,.07)}
.btn-resume{color:var(--gn);border-color:rgba(63,185,80,.4);background:rgba(63,185,80,.07)}
.btn-sq{color:var(--rd);border-color:rgba(248,81,73,.4);background:rgba(248,81,73,.07)}
.btn-pri{color:#fff;background:var(--ac);border-color:var(--ac)}
/* tables */
.tw{overflow-x:auto;border:1px solid var(--br);border-radius:8px;margin-bottom:20px}
table{width:100%;border-collapse:collapse}
th{background:var(--sf);color:var(--mu);font-size:11px;text-transform:uppercase;letter-spacing:.5px;
   padding:10px 12px;text-align:left;border-bottom:1px solid var(--br)}
td{padding:9px 12px;border-bottom:1px solid rgba(48,54,61,.5);white-space:nowrap}
tr:last-child td{border-bottom:none}
tbody tr:hover td{background:rgba(56,139,253,.04)}
.empty{text-align:center;color:var(--mu);padding:28px}
/* charts */
.cr2{display:grid;grid-template-columns:2fr 1fr 1fr;gap:14px;margin-bottom:20px}
.cb{background:var(--sf);border:1px solid var(--br);border-radius:8px;padding:16px}
.ct{font-size:11px;color:var(--mu);text-transform:uppercase;letter-spacing:.6px;margin-bottom:10px}
/* config */
.cg{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;margin-bottom:16px}
.cs2{grid-column:1/-1;font-size:11px;color:var(--mu);text-transform:uppercase;letter-spacing:1px;
     font-weight:600;padding-top:8px;border-top:1px solid var(--br)}
.fg{background:var(--sf);border:1px solid var(--br);border-radius:8px;padding:14px}
.fg label{display:block;font-size:11px;color:var(--mu);margin-bottom:6px;text-transform:uppercase;letter-spacing:.4px}
.fg input,.fg select{width:100%;background:var(--bg);border:1px solid var(--br);
  color:var(--tx);padding:7px 10px;border-radius:6px;font-size:14px}
.fg input:focus{outline:none;border-color:var(--ac)}
/* logs */
.lf{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.lfb{padding:4px 11px;border-radius:12px;border:1px solid var(--br);background:none;
     color:var(--mu);cursor:pointer;font-size:12px}
.lfb.active{border-color:var(--ac);color:var(--ac);background:rgba(56,139,253,.1)}
.lb{background:var(--sf);border:1px solid var(--br);border-radius:8px;height:500px;
    overflow-y:auto;font-family:'Consolas',monospace;font-size:12px}
.lr{display:flex;gap:10px;padding:5px 12px;border-bottom:1px solid rgba(48,54,61,.3)}
.lr:hover{background:rgba(255,255,255,.02)}
.lt{color:var(--mu);white-space:nowrap;min-width:135px;font-size:11px}
.lv{min-width:56px;font-weight:700;font-size:11px}
.lr.log-INFO .lv{color:var(--ac)}.lr.log-DEBUG .lv{color:var(--mu)}
.lr.log-WARNING .lv{color:var(--yw)}.lr.log-ERROR .lv,.lr.log-CRITICAL .lv{color:var(--rd)}
.lm{flex:1;word-break:break-word;line-height:1.5}
/* filters */
.fr{display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap;align-items:center}
.fr input,.fr select{background:var(--sf);border:1px solid var(--br);
  color:var(--tx);padding:6px 10px;border-radius:6px;font-size:13px}
.sh{font-size:12px;font-weight:600;color:var(--mu);margin-bottom:10px;
    text-transform:uppercase;letter-spacing:.5px}
@media(max-width:768px){.cr2{grid-template-columns:1fr}}
</style>
</head>
<body>

<header>
  <span class="logo">📈 Trading Bot</span>
  <span id="sBadge"  class="badge b-stopped">STOPPED</span>
  <span id="mBadge"  class="badge b-paper">PAPER</span>
  <span id="mkBadge" class="badge b-closed">MARKET CLOSED</span>
  <span class="sp"></span>
  <span class="ri">Auto-refresh 5s &nbsp;|&nbsp; <span id="lr">—</span></span>
</header>

<nav>
  <button class="tb active" onclick="showTab('status',this)">Live Status</button>
  <button class="tb"        onclick="showTab('history',this)">Trade History</button>
  <button class="tb"        onclick="showTab('config',this)">Strategy Config</button>
  <button class="tb"        onclick="showTab('logs',this)">Log Viewer</button>
</nav>

<div class="wrap">

<!-- ── Status ── -->
<div id="p-status" class="panel active">
  <div class="cards">
    <div class="card"><div class="cl">Today P&amp;L</div>
      <div class="cv neu" id="tPnl">₹0</div><div class="cs" id="tTrades">0 trades</div></div>
    <div class="card"><div class="cl">Win Rate</div>
      <div class="cv neu" id="tWR">—</div><div class="cs" id="tWL">0W / 0L</div></div>
    <div class="card"><div class="cl">Open Positions</div>
      <div class="cv neu" id="oCount">0</div><div class="cs" id="oSym">—</div></div>
    <div class="card"><div class="cl">Market</div>
      <div class="cv neu" id="mTime">—</div><div class="cs" id="mSub">—</div></div>
    <div class="card"><div class="cl">Strategy</div>
      <div class="cv neu" id="strat" style="font-size:15px">—</div>
      <div class="cs" id="stratSym">—</div></div>
  </div>
  <div class="ctrls">
    <button class="btn btn-pause"  onclick="ctrl('pause')">⏸ Pause Trading</button>
    <button class="btn btn-resume" onclick="ctrl('resume')">▶ Resume Trading</button>
    <button class="btn btn-sq"     onclick="sqOff()">🚨 Square Off All</button>
  </div>
  <div class="sh">Open Positions</div>
  <div class="tw"><table>
    <thead><tr><th>Symbol</th><th>Dir</th><th>Entry ₹</th><th>SL ₹</th>
      <th>Target ₹</th><th>Qty</th><th>Trail</th><th>Partial</th><th>Since</th></tr></thead>
    <tbody id="posT"><tr><td class="empty" colspan="9">No open positions</td></tr></tbody>
  </table></div>
</div>

<!-- ── History ── -->
<div id="p-history" class="panel">
  <div class="fr">
    <label style="color:var(--mu);font-size:12px">From</label><input type="date" id="fF">
    <label style="color:var(--mu);font-size:12px">To</label><input type="date" id="fT">
    <input type="text" id="fS" placeholder="Symbol" style="width:110px">
    <select id="fD"><option value="">All Dirs</option><option>LONG</option><option>SHORT</option></select>
    <button class="btn btn-pri" onclick="loadTrades()">Filter</button>
    <button class="btn" style="color:var(--mu);border-color:var(--br)" onclick="clrF()">Clear</button>
  </div>
  <div class="cr2">
    <div class="cb"><div class="ct">Equity Curve</div><canvas id="eqC" height="155"></canvas></div>
    <div class="cb"><div class="ct">Win / Loss</div><canvas id="wlC" height="155"></canvas></div>
    <div class="cb"><div class="ct">Daily P&amp;L — 30 Days</div><canvas id="dpC" height="155"></canvas></div>
  </div>
  <div class="sh" id="tHdr">All Trades</div>
  <div class="tw"><table>
    <thead><tr><th>#</th><th>Date</th><th>Symbol</th><th>Dir</th><th>Entry ₹</th>
      <th>Exit ₹</th><th>Qty</th><th>P&amp;L</th><th>Reason</th><th>Dur</th><th>Mode</th></tr></thead>
    <tbody id="tT"><tr><td class="empty" colspan="11">No trades yet</td></tr></tbody>
  </table></div>
</div>

<!-- ── Config ── -->
<div id="p-config" class="panel">
  <form id="cfgF" onsubmit="saveCfg(event)">
    <div class="cg" id="cfgG"><div style="color:var(--mu)">Loading…</div></div>
    <div style="display:flex;gap:10px;margin-top:8px">
      <button type="submit" class="btn btn-pri">Save Changes</button>
      <button type="button" class="btn" style="color:var(--mu);border-color:var(--br)" onclick="loadCfg()">Reset</button>
    </div>
  </form>
  <div id="cfgMsg" style="margin-top:12px;font-size:13px"></div>
</div>

<!-- ── Logs ── -->
<div id="p-logs" class="panel">
  <div class="lf">
    <button class="lfb active" onclick="setLL('ALL',this)">ALL</button>
    <button class="lfb" onclick="setLL('INFO',this)">INFO</button>
    <button class="lfb" onclick="setLL('WARNING',this)">WARNING</button>
    <button class="lfb" onclick="setLL('ERROR',this)">ERROR</button>
    <button class="lfb" onclick="setLL('DEBUG',this)">DEBUG</button>
    <button class="btn btn-pri" style="margin-left:auto;padding:4px 12px" onclick="loadLogs()">Refresh</button>
  </div>
  <div class="lb" id="logB"></div>
</div>

</div>

<script>
let _ll='ALL', _ch={}, _tab='status';

function showTab(n,btn){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tb').forEach(b=>b.classList.remove('active'));
  document.getElementById('p-'+n).classList.add('active');
  btn.classList.add('active');
  _tab=n;
  if(n==='history'){loadTrades();loadCharts();}
  if(n==='config'){loadCfg();}
  if(n==='logs'){loadLogs();}
}

async function api(u){try{const r=await fetch(u);return r.ok?await r.json():null;}catch{return null;}}
async function post(u,b){try{const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});return await r.json();}catch(e){return{ok:false,error:String(e)};}}

async function tick(){
  const d=await api('/api/status');
  if(d){
    const st=d.bot_status||'STOPPED';
    const sb=document.getElementById('sBadge');
    sb.textContent=st; sb.className='badge b-'+st.toLowerCase();
    const mb=document.getElementById('mBadge');
    mb.textContent=d.mode||'PAPER'; mb.className='badge b-'+(d.mode||'PAPER').toLowerCase();
    const mk=d.market||{};
    const mkb=document.getElementById('mkBadge');
    if(mk.is_open){mkb.textContent='🟢 MARKET OPEN';mkb.className='badge b-open';
      document.getElementById('mTime').textContent='OPEN';
      document.getElementById('mSub').textContent=mk.mins_to_close?`Closes in ${mk.mins_to_close} min`:'Open';}
    else{mkb.textContent='🔴 MARKET CLOSED';mkb.className='badge b-closed';
      document.getElementById('mTime').textContent='CLOSED';
      document.getElementById('mSub').textContent=mk.mins_to_open?`Opens in ${mk.mins_to_open} min`:'Weekend/Holiday';}
    const t=d.today||{};
    const pnl=t.net_pnl||0;
    const pe=document.getElementById('tPnl');
    pe.textContent=fmt(pnl); pe.className='cv '+(pnl>0?'pos':pnl<0?'neg':'neu');
    document.getElementById('tTrades').textContent=(t.total_trades||0)+' trades';
    document.getElementById('tWR').textContent=t.win_rate?t.win_rate+'%':'—';
    document.getElementById('tWL').textContent=(t.winning_trades||0)+'W / '+(t.losing_trades||0)+'L';
    document.getElementById('strat').textContent=d.strategy||'—';
    document.getElementById('stratSym').textContent=d.symbol||'—';
    document.getElementById('lr').textContent=new Date().toLocaleTimeString();
  }
  const ps=await api('/api/positions');
  const pb=document.getElementById('posT');
  if(!ps||!ps.length){pb.innerHTML='<tr><td class="empty" colspan="9">No open positions</td></tr>';
    document.getElementById('oCount').textContent='0';document.getElementById('oSym').textContent='—';}
  else{
    document.getElementById('oCount').textContent=ps.length;
    document.getElementById('oSym').textContent=ps.map(p=>p.symbol).join(', ');
    pb.innerHTML=ps.map(p=>`<tr>
      <td><b>${p.symbol}</b></td>
      <td><span class="badge ${p.direction==='LONG'?'b-open':'b-stopped'}">${p.direction}</span></td>
      <td>₹${(p.entry_price||0).toFixed(2)}</td><td>₹${(p.current_sl||0).toFixed(2)}</td>
      <td>₹${(p.target||0).toFixed(2)}</td><td>${p.quantity}</td>
      <td>${p.trail_active?'✅':'—'}</td><td>${p.partial_done?'✅':'—'}</td>
      <td style="color:var(--mu)">${p.entry_time?p.entry_time.slice(11,16):'—'}</td>
    </tr>`).join('');
  }
  if(_tab==='logs') loadLogs();
}

async function loadTrades(){
  const from=document.getElementById('fF').value,to=document.getElementById('fT').value,
        sym=document.getElementById('fS').value.trim().toUpperCase(),dir=document.getElementById('fD').value;
  let u='/api/trades?limit=200';
  if(from)u+='&from='+from;if(to)u+='&to='+to;if(sym)u+='&symbol='+sym;if(dir)u+='&direction='+dir;
  const rows=await api(u);
  const tb=document.getElementById('tT');
  if(!rows||!rows.length){tb.innerHTML='<tr><td class="empty" colspan="11">No trades found</td></tr>';
    document.getElementById('tHdr').textContent='No Trades';return;}
  document.getElementById('tHdr').textContent=rows.length+' Trades';
  tb.innerHTML=rows.map((t,i)=>{
    const p=t.pnl,pc=p>0?'pos':p<0?'neg':'neu',
          d=dur(t.entry_time,t.exit_time);
    return`<tr>
      <td style="color:var(--mu)">${i+1}</td><td>${t.date}</td><td><b>${t.symbol}</b></td>
      <td><span class="badge ${t.direction==='LONG'?'b-open':'b-stopped'}">${t.direction}</span></td>
      <td>₹${(t.entry_price||0).toFixed(2)}</td><td>₹${(t.exit_price||0).toFixed(2)}</td>
      <td>${t.quantity}</td><td class="${pc}"><b>${fmt(p)}</b></td>
      <td style="color:var(--mu);font-size:12px">${t.exit_reason||'—'}</td>
      <td style="color:var(--mu)">${d}</td>
      <td>${t.is_paper?'<span class="badge b-paper">PAPER</span>':'<span class="badge b-live">LIVE</span>'}</td>
    </tr>`;}).join('');
}

function clrF(){['fF','fT','fS'].forEach(id=>document.getElementById(id).value='');
  document.getElementById('fD').value='';loadTrades();}

async function loadCharts(){
  const [eq,dp,tr]=await Promise.all([api('/api/equity-curve'),api('/api/daily-pnl'),api('/api/trades?limit=500')]);
  drawEq(eq||[]);drawDP(dp||[]);drawWL(tr||[]);
}

function drawEq(pts){
  const c=document.getElementById('eqC').getContext('2d');
  if(_ch.eq)_ch.eq.destroy();
  _ch.eq=new Chart(c,{type:'line',data:{
    labels:pts.map(p=>p.time.slice(0,10)),
    datasets:[{data:pts.map(p=>p.equity),borderColor:'#388bfd',backgroundColor:'rgba(56,139,253,.07)',
      borderWidth:2,pointRadius:0,fill:true,tension:.3}]},
    options:{responsive:true,plugins:{legend:{display:false}},
      scales:{x:{ticks:{color:'#8b949e',maxTicksLimit:6,font:{size:10}},grid:{color:'rgba(48,54,61,.5)'}},
              y:{ticks:{color:'#8b949e',callback:v=>'₹'+v.toLocaleString('en-IN'),font:{size:10}},grid:{color:'rgba(48,54,61,.5)'}}}}});
}

function drawWL(tr){
  const w=tr.filter(t=>t.pnl>0).length,l=tr.filter(t=>t.pnl<=0).length;
  const c=document.getElementById('wlC').getContext('2d');
  if(_ch.wl)_ch.wl.destroy();
  _ch.wl=new Chart(c,{type:'doughnut',data:{
    labels:['Wins','Losses'],
    datasets:[{data:[w,l],backgroundColor:['rgba(63,185,80,.7)','rgba(248,81,73,.7)'],
      borderColor:['#3fb950','#f85149'],borderWidth:2}]},
    options:{responsive:true,plugins:{legend:{position:'bottom',labels:{color:'#8b949e',font:{size:11}}}}}});
}

function drawDP(dp){
  const c=document.getElementById('dpC').getContext('2d');
  if(_ch.dp)_ch.dp.destroy();
  _ch.dp=new Chart(c,{type:'bar',data:{
    labels:dp.map(d=>d.date.slice(5)),
    datasets:[{data:dp.map(d=>d.pnl),
      backgroundColor:dp.map(d=>d.pnl>=0?'rgba(63,185,80,.55)':'rgba(248,81,73,.55)'),
      borderColor:dp.map(d=>d.pnl>=0?'#3fb950':'#f85149'),borderWidth:1}]},
    options:{responsive:true,plugins:{legend:{display:false}},
      scales:{x:{ticks:{color:'#8b949e',maxTicksLimit:12,font:{size:9}},grid:{display:false}},
              y:{ticks:{color:'#8b949e',callback:v=>'₹'+v.toLocaleString('en-IN'),font:{size:10}},grid:{color:'rgba(48,54,61,.5)'}}}}});
}

const _CFG={
  volume_surge_multiplier:{l:'Volume Surge Multiplier',s:.1,mn:1,mx:5},
  rsi_long_min:{l:'RSI Long Min',s:1,mn:20,mx:60},rsi_long_max:{l:'RSI Long Max',s:1,mn:50,mx:80},
  rsi_short_min:{l:'RSI Short Min',s:1,mn:20,mx:60},rsi_short_max:{l:'RSI Short Max',s:1,mn:40,mx:80},
  rsi_overbought:{l:'RSI Overbought',s:1,mn:60,mx:90},rsi_oversold:{l:'RSI Oversold',s:1,mn:10,mx:40},
  sl_atr_multiplier:{l:'SL ATR Multiplier',s:.1,mn:.5,mx:4},rr_ratio:{l:'Reward:Risk Ratio',s:.1,mn:1,mx:5},
  ema_period:{l:'EMA Period',s:1,mn:3,mx:50},rsi_period:{l:'RSI Period',s:1,mn:5,mx:30},
  atr_period:{l:'ATR Period',s:1,mn:5,mx:30},volume_ma_period:{l:'Volume MA Period',s:1,mn:5,mx:50},
};

async function loadCfg(){
  const d=await api('/api/config');if(!d)return;
  const g=document.getElementById('cfgG');const p=d.strategy_params||{};
  let h='<div class="cs2">Strategy Parameters — live editable</div>';
  for(const[k,v]of Object.entries(p)){
    const m=_CFG[k]||{l:k,s:'any',mn:'',mx:''};
    h+=`<div class="fg"><label>${m.l}</label>
      <input name="${k}" type="number" step="${m.s}" min="${m.mn}" max="${m.mx}" value="${v}"></div>`;
  }
  h+='<div class="cs2">Global Settings (read-only)</div>';
  [['Capital','₹'+(d.capital||0).toLocaleString('en-IN')],['Risk/Trade',d.risk_per_trade+'%'],
   ['Max Daily Loss','₹'+(d.max_daily_loss||0).toLocaleString('en-IN')],
   ['Max Trades/Day',d.max_trades_day],['Paper Mode',d.paper_trade?'Yes':'No']]
   .forEach(([l,v])=>h+=`<div class="fg"><label>${l}</label><input type="text" value="${v}" readonly style="opacity:.55;cursor:default"></div>`);
  g.innerHTML=h;document.getElementById('cfgMsg').textContent='';
}

async function saveCfg(e){
  e.preventDefault();
  const body={};
  document.getElementById('cfgF').querySelectorAll('input[name]:not([readonly])')
    .forEach(i=>body[i.name]=parseFloat(i.value));
  const r=await post('/api/config',body);
  const m=document.getElementById('cfgMsg');
  if(r.updated){m.style.color='var(--gn)';
    m.textContent='✓ Saved: '+r.updated.map(c=>c.param+' → '+c.new).join(' | ');}
  else{m.style.color='var(--rd)';m.textContent='✗ '+(r.error||'Save failed');}
}

function setLL(l,btn){_ll=l;document.querySelectorAll('.lfb').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');loadLogs();}

async function loadLogs(){
  const logs=await api('/api/logs?level='+_ll+'&limit=100');
  const b=document.getElementById('logB');
  if(!logs||!logs.length){b.innerHTML='<div style="color:var(--mu);padding:28px;text-align:center">No entries</div>';return;}
  b.innerHTML=logs.map(l=>`<div class="lr log-${l.level}">
    <span class="lt">${l.time.slice(0,19).replace('T',' ')}</span>
    <span class="lv">${l.level}</span>
    <span class="lm">${esc(l.message)}</span></div>`).join('');
  b.scrollTop=b.scrollHeight;
}

async function ctrl(a){const r=await post('/api/control',{action:a});if(r.ok)tick();else alert('Error: '+(r.error||a+' failed'));}
function sqOff(){if(confirm('Square off ALL open positions? This cannot be undone.'))ctrl('squareoff');}

function fmt(v){if(v==null)return'—';const s=v>=0?'+₹':'-₹';return s+Math.abs(v).toLocaleString('en-IN',{minimumFractionDigits:2,maximumFractionDigits:2});}
function dur(a,b){if(!a||!b)return'—';const m=Math.round((new Date(b)-new Date(a))/60000);return m<60?m+'m':Math.floor(m/60)+'h '+(m%60)+'m';}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

tick();
setInterval(tick,5000);
</script>
</body>
</html>"""
