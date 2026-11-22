// @vitest-environment jsdom
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { api, type MonitorRule } from '@/lib/api'
import { RuleEditor } from './RuleEditor'

vi.mock('@/lib/useSharedQueries', () => ({
  usePreferences: () => ({ data: {} }), useQuoteStatus: () => ({ data: {} }),
}))
vi.mock('@/components/screener/SignalPicker', () => ({ SignalPicker: () => null }))

let host: HTMLDivElement
let root: ReturnType<typeof createRoot>
let client: QueryClient
const rule: MonitorRule = {
  id: 'demo_rule', name: '示例监控', enabled: true, type: 'strategy', asset_type: 'stock',
  scope: 'all', symbols: [], strategy_id: 'demo', direction: 'entry',
  notify_events: ['pool_entry'], conditions: [], logic: 'and', cooldown_seconds: 3600,
  severity: 'info', message: '',
}

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  host = document.createElement('div'); document.body.append(host)
  root = createRoot(host)
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  vi.spyOn(api, 'monitorRuleOptions').mockResolvedValue({} as Awaited<ReturnType<typeof api.monitorRuleOptions>>)
  const strategies = { presets: [], load_errors: [], legacy_migration: undefined }
  vi.spyOn(api, 'screenerStrategies').mockResolvedValue(strategies)
  vi.spyOn(api, 'monitorRuleSave').mockImplementation(async saved => ({ ok: true, rule: saved }))
})
afterEach(async () => {
  await act(async () => root.unmount())
  host.remove(); client.clear(); vi.restoreAllMocks()
})

async function settle() {
  await act(async () => { await new Promise(resolve => setTimeout(resolve, 15)) })
}
async function render(current: MonitorRule | null, key = 'edit') {
  await act(async () => root.render(
    <MemoryRouter><QueryClientProvider client={client}>
      <RuleEditor key={key} rule={current} onClose={() => {}} />
    </QueryClientProvider></MemoryRouter>,
  ))
  await settle()
}
function checkbox() {
  const label = Array.from(host.querySelectorAll('label')).find(el => el.textContent?.includes('自动加入自选，按策略分组'))
  return label?.querySelector<HTMLInputElement>('input')
}
function save() { return Array.from(host.querySelectorAll('button')).find(el => el.textContent === '保存')! }

it('defaults historical rules to off, saves opt-in and preserves it when reopened', async () => {
  await render(rule)
  expect(checkbox()?.checked).toBe(false)
  expect(host.textContent).toContain('下一轮监控补入当前结果池')
  await act(async () => checkbox()!.click())
  await act(async () => save().click())
  await settle()
  expect(api.monitorRuleSave).toHaveBeenCalledWith(expect.objectContaining({ auto_watchlist: true }))
  await render({ ...rule, auto_watchlist: true }, 'reopened')
  expect(checkbox()?.checked).toBe(true)
  await render(null, 'new')
  expect(checkbox()).toBeUndefined()
})

it('hides the option for ETFs and removes stale opt-in fields before saving', async () => {
  await render({ ...rule, asset_type: 'etf', auto_watchlist: true })
  expect(checkbox()).toBeUndefined()
  await act(async () => save().click())
  await settle()
  const saved = vi.mocked(api.monitorRuleSave).mock.calls[0][0]
  expect(saved).not.toHaveProperty('auto_watchlist')
})

it('keeps the checked option and reports save failures without closing', async () => {
  vi.mocked(api.monitorRuleSave).mockRejectedValue(new Error('保存失败，请重试'))
  await render({ ...rule, auto_watchlist: true })
  await act(async () => save().click())
  await settle()
  expect(host.textContent).toContain('保存失败，请重试')
  expect(checkbox()?.checked).toBe(true)
  expect(save().disabled).toBe(false)
})
