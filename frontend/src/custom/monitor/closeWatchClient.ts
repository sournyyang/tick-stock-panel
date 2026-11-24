import { ApiError } from '@/lib/api'

export interface SignalTrack { id: string; symbol: string; name: string; signal_date: string }
export interface DingTalkStatus {
  enabled: boolean; configured: boolean; status: string; message: string; pending: number
  records: Array<{ id: string; status: string; detail: string; queued_at: number; symbol: string; phase: string; type: string; elapsed_ms?: number }>
}
export interface CloseWatchState {
  dingtalk?: DingTalkStatus
  watchlist?: { enabled: boolean; message: string; warning: string | null }
  enabled: boolean
  status: string
  message: string
  phase: 'preview' | 'confirmed' | null
  binding: { strategy_id: string; name: string; revision: string; start_time: string; max_hold_days: number | null } | null
  tracks: SignalTrack[]
  candidates: Array<{ symbol: string; name: string; price: number; signal_date: string; floor_price?: number | null; floor_signal_date?: string; note?: string }>
  watch_status: Array<SignalTrack & { price: number | null; floor_price: number | null; signal_age: number | null; reasons: string[]; message: string; ended: boolean }>
  checked_at?: string
  as_of?: string
  coverage?: { observed_symbols: number; reported_symbols?: number }
  events: Array<{ id: string; ts: number; day: string; phase: string; symbol: string; name: string; type: string; message: string }>
}

export const closeWatchKey = ['custom', 'close-watch', 'state'] as const

async function request<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), 15_000)
  try {
    const response = await fetch(`/api/custom/close-watch/${path}`, {
      method, headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal,
    })
    const payload = await response.json().catch(() => null)
    if (!response.ok) {
      const detail = payload?.detail
      const message = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(x => x.msg).join('；') : `尾盘监控请求失败 (${response.status})`
      throw new ApiError(response.status === 401 ? `未登录 (401)：${message}` : message, response.status)
    }
    return payload as T
  } catch (error) {
    if (controller.signal.aborted) throw new Error('尾盘监控请求超时，请刷新状态核对结果')
    throw error
  } finally {
    window.clearTimeout(timer)
  }
}

export const closeWatchApi = {
  configureDingTalk: (enabled: boolean) => request<CloseWatchState>('dingtalk', 'PUT', { enabled }),
  testDingTalk: () => request<CloseWatchState>('dingtalk/test', 'POST'),
  state: () => request<CloseWatchState>('state'),
  configure: (strategy_id: string, enabled: boolean, start_time: string) => request<CloseWatchState>('config', 'PUT', { strategy_id, enabled, start_time }),
}
