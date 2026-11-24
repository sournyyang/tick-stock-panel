import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Clock3, RefreshCw } from 'lucide-react'
import { api } from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import { fmtPrice } from '@/lib/format'
import { selectableStrategies } from '@/lib/strategyLibrary'
import { closeWatchApi, closeWatchKey, type CloseWatchState } from './closeWatchClient'

const input = 'rounded-btn border border-border bg-base px-2 py-1.5 text-xs text-foreground min-w-0'
const deliveryLabels: Record<string, string> = { queued: '等待发送', sending: '正在发送', sent: '钉钉已接收', failed: '发送失败', expired: '已过时未发送', cancelled: '已取消发送', unknown: '发送结果未知' }
const button = 'rounded-btn border border-border bg-elevated px-2 py-1.5 text-xs disabled:opacity-40'

export function CloseWatchPanel() {
  const qc = useQueryClient()
  const query = useQuery({ queryKey: closeWatchKey, queryFn: closeWatchApi.state, refetchInterval: 15_000, retry: false })
  const strategies = useQuery({ queryKey: QK.screenerStrategies('stock'), queryFn: () => api.screenerStrategies('stock') })
  const state = query.data
  const [open, setOpen] = useState(false)
  const [strategy, setStrategy] = useState('')
  const [start, setStart] = useState('')
  const sid = strategy || state?.binding?.strategy_id || 'custom_triple_volume_two60_body25'
  const time = start || state?.binding?.start_time || '14:50'
  const saved = (value: CloseWatchState) => { qc.setQueryData(closeWatchKey, value) }
  const config = useMutation({ mutationFn: (enabled: boolean) => closeWatchApi.configure(sid, enabled, time), onSuccess: saved })
  const pause = useMutation({ mutationFn: () => closeWatchApi.configure(state!.binding!.strategy_id, false, state!.binding!.start_time), onSuccess: saved })
  const dingtalk = useMutation({ mutationFn: (enabled: boolean) => closeWatchApi.configureDingTalk(enabled), onSuccess: saved })
  const test = useMutation({ mutationFn: () => closeWatchApi.testDingTalk(), onSuccess: saved })
  const error = query.error || config.error || pause.error || strategies.error || dingtalk.error || test.error
  const deliveryBusy = dingtalk.isPending || test.isPending
  const busy = config.isPending || pause.isPending
  return <section className="mx-auto w-full max-w-7xl rounded-xl border border-accent/25 bg-surface/60" aria-label="尾盘策略监控">
    <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-3">
      <button type="button" className="flex min-w-0 items-center gap-2 text-left" aria-expanded={open} onClick={() => setOpen(!open)}>
        <Clock3 className="h-4 w-4 shrink-0 text-accent" /><span className="text-sm font-medium">尾盘策略监控</span>
        <span className="text-[11px] text-muted">{state?.status === 'blocked' ? '需要核对' : state?.enabled ? `${state.binding?.start_time}当天预警` : '未启用'} · 信号观察 {state?.tracks.length ?? 0}</span>
      </button>
      <div className="flex items-center gap-2"><span className="max-w-md text-[11px] text-muted">{state?.message || (query.isPending ? '读取状态…' : '')}</span>
        <button type="button" className={button} aria-label="刷新尾盘监控" onClick={() => void query.refetch()}><RefreshCw className="h-3 w-3" /></button>
        <button type="button" className={button} onClick={() => setOpen(!open)}>{open ? '收起' : '配置与信号跟踪'}</button>
      </div>
    </div>
    {error && <p role="alert" className="px-4 pb-3 text-xs text-danger">{error.message}</p>}
    {open && <div className="max-h-[65vh] space-y-4 overflow-y-auto border-t border-border/60 px-4 py-3">
      <p className="text-[11px] leading-relaxed text-secondary">买入和卖出信号均在尾盘当天提示，不等第二天。自动跟踪本策略买点之后的卖点，不代表已经买入或卖出；无需模拟账户和持仓登记，不代下单。</p>
      <p className="text-[11px] leading-relaxed text-muted">尾盘预警尚可能变化，收盘后另给确认或取消。历史回测按次日开盘成交，尾盘当天成交的效果尚未单独验证。可在下方启用钉钉，接收新预警、收盘确认和取消通知。</p>
      <p className="text-[11px] text-muted">原有普通监控规则的通知独立运行；自动加入分组自选需保持同策略普通规则启用，并开启“自动加入自选”。</p>
      {state?.watchlist && <p className="text-[11px] text-muted">分组自选：{state.watchlist.message}。跟随同策略普通规则的“自动加入自选”开关；尾盘临时预警不自动加入。</p>}
      {state?.watchlist?.warning && <p role="alert" className="text-xs text-danger">{state.watchlist.warning}</p>}
      <div className="flex flex-wrap items-end gap-2">
        <label className="flex min-w-48 flex-1 flex-col gap-1 text-[11px] text-muted">尾盘策略<select aria-label="尾盘策略" className={input} value={sid} onChange={e => setStrategy(e.target.value)} disabled={busy}>
          {selectableStrategies(strategies.data?.presets ?? [], [sid]).filter(s => s.execution_backend === 'matrix_native' && s.timeframes?.includes('1d')).map(s => <option value={s.id} key={s.id}>{s.name}</option>)}
        </select></label>
        <label className="flex flex-col gap-1 text-[11px] text-muted">预警开始<input aria-label="尾盘预警开始" type="time" min="14:30" max="14:59" value={time} onChange={e => setStart(e.target.value)} className={input} disabled={busy} /></label>
        <button type="button" className={`${button} text-accent`} disabled={busy || query.isPending || query.isError} onClick={() => config.mutate(true)}>{config.isPending ? '保存中…' : '保存并启用'}</button>
        <button type="button" className={button} disabled={busy || !state?.enabled} onClick={() => pause.mutate()}>暂停</button>
      </div>
      {state?.binding && <p className="text-[11px] text-muted">当前绑定：{state.binding.name} · {state.binding.max_hold_days == null ? '无固定观察期限' : `买点后${state.binding.max_hold_days}个市场交易日到期观察`}。更换策略会重建信号观察，不影响实际账户。{state.checked_at && ` 最近检查：${state.checked_at.slice(0,19).replace('T',' ')}`}</p>}
      {state?.dingtalk && <div className="space-y-2 rounded-btn border border-border p-3" aria-label="尾盘钉钉推送">
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-xs font-medium"><input type="checkbox" aria-label="启用尾盘钉钉推送" checked={state.dingtalk.enabled}
            disabled={deliveryBusy || state.dingtalk.status === 'blocked' || (!state.dingtalk.enabled && !state.dingtalk.configured)} onChange={e => dingtalk.mutate(e.target.checked)} />尾盘钉钉推送</label>
          <button type="button" className={button} disabled={deliveryBusy || !state.dingtalk.enabled || !state.dingtalk.configured || state.dingtalk.status === 'blocked'} onClick={() => test.mutate()}>发送连接测试</button>
          <span className="text-[11px] text-muted">{state.dingtalk.message} · 待发送 {state.dingtalk.pending}</span>
        </div>
        <p className="text-[11px] text-muted">复用设置中的机器人和加签配置；仅推送启用后的新信号，不补发历史。发送在后台进行，不影响信号计算。收盘确认仅供复核，当日已无法交易。</p>
        {!state.dingtalk.configured && <p className="text-[11px] text-warning">请先到设置页面配置钉钉机器人。</p>}
        <details><summary className="cursor-pointer text-[11px] text-muted">最近钉钉发送记录 ({state.dingtalk.records.length})</summary>
          <p className="my-2 text-[11px] text-muted">网络失败或中断后不自动重复发送，避免重复提醒；发送结果未知时请核对群消息。</p>
          {state.dingtalk.records.slice(0, 12).map(r => <p key={r.id} className="mt-2 text-[11px] leading-relaxed">
            {new Date(r.queued_at).toLocaleTimeString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false })} · {r.phase === 'test' ? '连接测试' : `${r.symbol} ${r.phase === 'preview' ? '尾盘预警' : '收盘复核'}`} · <span className={['failed', 'unknown'].includes(r.status) ? 'text-danger' : 'text-muted'}>{deliveryLabels[r.status] || r.status}</span> · {r.detail}{r.elapsed_ms != null && ` · ${r.elapsed_ms}毫秒`}
          </p>)}
        </details>
      </div>}
      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2"><div className="text-xs font-medium">最近检查的买入候选{state?.as_of && <span className="ml-2 text-muted">{state.as_of}</span>}</div>
          {!state?.candidates.length && <p className="text-[11px] text-muted">尚无可展示候选；请结合上方行情状态判断，旧日数据不会当作今日信号。</p>}
          {state?.candidates.map(c => <div key={c.symbol} className="rounded-btn border border-border p-2 text-[11px]">{c.name} {c.symbol} · {fmtPrice(c.price)} · 信号日{c.signal_date}<p className="mt-1 text-muted">防守参考价：{c.floor_price == null ? '待核对' : `${fmtPrice(c.floor_price)}元`} · 收盘失守条件，非保证成交价{c.floor_signal_date && ` · 原买点${c.floor_signal_date}`}</p>{c.floor_price != null && c.price < c.floor_price && <p className="mt-1 text-warning">参考价已低于防守线，不能视为正常的下方止损价</p>}{c.note && <p className="mt-1 text-muted">{c.note}</p>}</div>)}
        </div>
        <div className="space-y-2"><div className="text-xs font-medium">买点后的自动信号观察</div>
          <p className="text-[11px] text-muted">只跟踪买点和卖点，不计算实际仓位或资金。首次补入可用历史中尚未出现卖点的观察，不补发旧日提醒。</p>
          {!state?.tracks.length && !state?.watch_status.length && <p className="text-[11px] text-muted">暂无活动观察，出现收盘确认买点后自动加入。</p>}
          {state?.watch_status.map(t => <div key={t.id} className="rounded-btn border border-border p-2 text-[11px]">
            <span className="font-medium">{t.name} {t.symbol}</span> · 原买点{t.signal_date}
            <p className="mt-1 text-muted">买点后{t.signal_age ?? '未知'}日 · 原防守线{t.floor_price == null ? '待计算' : fmtPrice(t.floor_price)}</p>
            {t.floor_price != null && t.price != null && t.price < t.floor_price && <p className="mt-1 text-warning">参考价已低于防守线，不能视为正常的下方止损价</p>}
            <p className={t.reasons.length ? 'mt-1 text-warning' : 'mt-1 text-muted'}>{t.reasons.join('；') || t.message}</p>
            {t.ended && <p className="mt-1 text-muted">卖点已确认，本次观察结束；不代表已实际卖出。</p>}
          </div>)}
        </div>
      </div>
      <details><summary className="cursor-pointer text-xs text-muted">最近尾盘记录 ({state?.events.length ?? 0})</summary><div className="mt-2 space-y-2">
        {!state?.events.length && <p className="text-[11px] text-muted">暂无记录。预警与确认分别去重，重启后继续去重。</p>}
        {state?.events.slice(0,12).map(e => <p key={e.id} className="rounded-btn border border-border p-2 text-[11px] leading-relaxed text-secondary">{e.day} {new Date(e.ts).toLocaleTimeString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false })} · {e.phase === 'preview' ? '当天预警' : '收盘确认'} · {e.message}</p>)}
      </div></details>
    </div>}
  </section>
}
