// @vitest-environment jsdom
import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { CloseWatchPanel } from './CloseWatchPanel'
import { closeWatchApi, type CloseWatchState } from './closeWatchClient'

vi.mock('@/lib/api', () => ({ api: { screenerStrategies: vi.fn(async () => ({ presets: [
  { id: 'demo', name: '已绑定策略', execution_backend: 'matrix_native', timeframes: ['1d'] },
  { id: 'other', name: '其他策略', execution_backend: 'matrix_native', timeframes: ['1d'] },
] })) } }))
vi.mock('./closeWatchClient', () => ({ closeWatchKey: ['custom', 'close-watch', 'state'], closeWatchApi: {
  state: vi.fn(), configure: vi.fn(),
} }))

const initial: CloseWatchState = {
  enabled: true, status: 'waiting', message: '等待交易日尾盘时段', phase: null,
  binding: { strategy_id: 'demo', name: '已绑定策略', revision: 'abc', start_time: '14:50', max_hold_days: 8 },
  tracks: [], candidates: [], watch_status: [], events: [],
}
let host: HTMLDivElement
let root: Root
let qc: QueryClient
beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  vi.mocked(closeWatchApi.state).mockResolvedValue(initial)
  vi.mocked(closeWatchApi.configure).mockResolvedValue(initial)
  host = document.createElement('div'); document.body.append(host)
  root = createRoot(host)
  qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
})
afterEach(async () => { await act(async () => root.unmount()); qc.clear(); host.remove(); vi.clearAllMocks() })
async function flush() { await act(async () => { await new Promise(r => setTimeout(r, 30)) }) }
async function click(text: string) {
  const button = [...host.querySelectorAll('button')].find(b => b.textContent === text)!
  await act(async () => button.click()); await flush()
}
async function render() {
  await act(async () => root.render(<QueryClientProvider client={qc}><CloseWatchPanel /></QueryClientProvider>))
  await flush(); await click('配置与信号跟踪')
}
it('announces same-day signals without account registration or assuming fills', async () => {
  await render()
  expect(host.textContent).toContain('不等第二天')
  expect(host.textContent).toContain('不代表已经买入或卖出')
  expect(host.textContent).toContain('无需模拟账户和持仓登记')
  expect(host.querySelector('input[type="date"]')).toBeNull()
  await click('保存并启用')
  expect(closeWatchApi.configure).toHaveBeenCalledWith('demo', true, '14:50')
})

it('shows the original signal and confirmed end as observations only', async () => {
  vi.mocked(closeWatchApi.state).mockResolvedValueOnce({ ...initial, watch_status: [{
    id: 't', symbol: '600000.SH', name: '示例股票', signal_date: '2026-11-18',
    price: 9.8, floor_price: 10, signal_age: 2, reasons: ['低于原买点绑定防守线'], ended: true, message: '',
  }] })
  await render()
  expect(host.textContent).toContain('原买点2026-11-18')
  expect(host.textContent).toContain('低于原买点绑定防守线')
  expect(host.textContent).toContain('不代表已实际卖出')
})

it('pause preserves the saved strategy even when the draft selection changed', async () => {
  await render()
  const select = host.querySelector('select[aria-label="尾盘策略"]') as HTMLSelectElement
  await act(async () => { select.value = 'other'; select.dispatchEvent(new Event('change', { bubbles: true })) })
  await click('暂停')
  expect(closeWatchApi.configure).toHaveBeenCalledWith('demo', false, '14:50')
})

it('does not allow configuration while the backend is unavailable', async () => {
  vi.mocked(closeWatchApi.state).mockRejectedValue(new Error('尾盘监控未就绪'))
  await render()
  expect(host.querySelector('[role="alert"]')?.textContent).toContain('尾盘监控未就绪')
  const button = [...host.querySelectorAll('button')].find(b => b.textContent === '保存并启用')!
  expect(button.disabled).toBe(true)
})
