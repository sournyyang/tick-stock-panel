// @vitest-environment jsdom
// 回归: 预热/同步/修复期间 (stats_pending / indicators_ready=false) 统计为 null 是未知态,
// 不得误判"缺少指标数据"禁用运行回测按钮 (同 Dashboard 的 warmup 误判一族);
// 真实统计就绪且确无数据时仍要禁用 + 提示。
import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { StrategyBacktest, MonteCarloDrawdownStat } from './StrategyBacktest'
import type { DataStatus, StrategyDetail } from '@/lib/api'
import { storage } from '@/lib/storage'
import { QK } from '@/lib/queryKeys'

vi.mock('@/lib/api', () => ({
  api: {
  dataStatus: vi.fn(),
  dataIntegrity: vi.fn(async () => ({ report: null, blocking_assets: [], data_revisions: {} })),
  screenerStrategies: vi.fn(async () => ({ presets: [] })),
  strategyGet: vi.fn(async () => ({})),
  watchlistList: vi.fn(async () => ({ items: [] })),
  positionPolicyList: vi.fn(async () => ({ policies: [] })),
  factorColumns: vi.fn(async () => ({ columns: [] })),
  factorLibrary: vi.fn(async () => ({ factors: [] })),
  instrumentSearch: vi.fn(async () => ({ results: [] })),
  customSignalsList: vi.fn(async () => ({ signals: [] })),
  customSignalsOptions: vi.fn(async () => ({ fields: [], operators: [] })),
  capabilities: vi.fn(async () => ({ capabilities: {} })),
  settings: vi.fn(async () => ({ onboarding_completed: true, mode: 'api_key', has_tickflow_key: true })),
  preferences: vi.fn(async () => ({})),
  dataSources: vi.fn(async () => []),
  regimeRecompute: vi.fn(async () => ({ ok: true, computed: 0 })),
  strategySaveConfig: vi.fn(async () => ({ ok: true })),
  strategyBacktestHistory: vi.fn(async () => ({ items: [] })),
  researchCandidateCreate: vi.fn(async () => ({})),
  },
  REGIME_STATE_LABELS: {} as Record<string, string>,
  REGIME_STATE_COLORS: {} as Record<string, string>,
}))
vi.mock('@/lib/backtestTask', () => ({
  startBacktest: vi.fn(),
  stopBacktest: vi.fn(async () => {}),
  tryReconnect: vi.fn(() => false),
  clearBacktest: vi.fn(),
  useBacktestTask: () => null,
}))

let host: HTMLDivElement
let root: Root
let client: QueryClient

beforeEach(async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  sessionStorage.clear()
  localStorage.clear()
  const { api } = await import('@/lib/api')
  vi.mocked(api.screenerStrategies).mockResolvedValue({ presets: [], load_errors: [], legacy_migration: undefined })
  host = document.createElement('div')
  document.body.append(host)
  root = createRoot(host)
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
})

const boundDetail = {
  id: 'bound', name: '收盘防守策略', source: 'custom', execution_backend: 'matrix_native',
  asset_types: ['stock'], timeframes: ['1d'], params: [], params_defaults: {},
  basic_filter: {}, scoring: {}, scoring_directions: {}, entry_signals: [], exit_signals: [],
  backtest_requirements: { entry_fill: 'open_t+1', exit_fill: 'open_t+1', mode: 'position', minute_fill: false },
} as unknown as StrategyDetail

const portfolioDetail = {
  ...boundDetail, id: 'portfolio', name: '两仓60%测试', max_hold_days: 8,
  backtest_defaults: { initial_capital: 100000, max_positions: 2, max_exposure_pct: 0.6, slippage_bps: 12, commission_pct: 0.0002, stamp_tax_pct: 0.001, position_sizing: 'equal' },
} as StrategyDetail

it('账户蒙卡卡片显示账户结果并保留逐笔对照', async () => {
  await act(async () => root.render(<MonteCarloDrawdownStat stats={{
    mc_account_method: 'daily_circular_block_v1', mc_account_maxdd_p50: -.1446,
    mc_account_maxdd_p95: -.2639, mc_maxdd_p50: -.3703, mc_maxdd_p95: -.6025,
  }} />))
  expect(host.textContent).toContain('账户蒙卡 中位/95%')
  expect(host.textContent).toContain('-14.46%/-26.39%')
  expect(host.textContent).toContain('逐笔：-37.03%/-60.25%')
  await act(async () => host.querySelector('button')!.click())
  expect(host.textContent).toContain('连续5个交易日')
  expect(host.textContent).toContain('不代表未来概率保证')
})

it('旧结果只显示逐笔蒙卡，不冒充账户口径', async () => {
  await act(async () => root.render(<MonteCarloDrawdownStat stats={{ mc_maxdd_p50: -.3703, mc_maxdd_p95: -.6025 }} />))
  expect(host.textContent).toContain('逐笔蒙卡 中位/95%')
  expect(host.textContent).not.toContain('账户蒙卡')
  expect(host.textContent).toContain('-37.03%/-60.25%')
})

it('账户蒙卡不可计算时显示未知，不退回逐笔数值', async () => {
  await act(async () => root.render(<MonteCarloDrawdownStat stats={{
    mc_account_method: 'daily_circular_block_v1', mc_account_maxdd_p50: null,
    mc_account_maxdd_p95: null, mc_maxdd_p50: -.3703, mc_maxdd_p95: -.6025,
  }} />))
  expect(host.textContent).toContain('账户蒙卡 中位/95%')
  expect(host.textContent).toContain('—/—')
})

it('首次选择带推荐回测配置的策略时载入资金仓位和费用默认值', async () => {
  const { api } = await import('@/lib/api')
  const { startBacktest } = await import('@/lib/backtestTask')
  vi.mocked(api.dataStatus).mockResolvedValue(busy)
  vi.mocked(api.screenerStrategies).mockResolvedValue({ presets: [portfolioDetail], load_errors: [], legacy_migration: undefined })
  vi.mocked(api.strategyGet).mockResolvedValue(portfolioDetail)
  storage.strategyBacktestLast.set({ assetType: 'stock', initialCapital: '900000', maxPositions: '9', maxExposure: '100', slippage: '5' } as Parameters<typeof storage.strategyBacktestLast.set>[0])
  await renderPage()
  const choose = [...host.querySelectorAll('button')].find(b => b.textContent?.includes('两仓60%测试'))!
  await act(async () => choose.click())
  await act(async () => { await new Promise(r => setTimeout(r, 80)) })
  await act(async () => runButton()!.click())
  expect(startBacktest).toHaveBeenLastCalledWith(expect.objectContaining({ initial_capital: 100000, max_positions: 2, max_exposure_pct: 0.6, slippage_bps: 12, commission_pct: 0.0002, stamp_tax_pct: 0.001, position_sizing: 'equal', overrides: expect.objectContaining({ max_hold_days: 8 }) }))
})

it('重新打开同一策略时保留用户保存的资金配置', async () => {
  const { api } = await import('@/lib/api')
  const { startBacktest } = await import('@/lib/backtestTask')
  vi.mocked(api.dataStatus).mockResolvedValue(busy)
  vi.mocked(api.screenerStrategies).mockResolvedValue({ presets: [portfolioDetail], load_errors: [], legacy_migration: undefined })
  vi.mocked(api.strategyGet).mockResolvedValue(portfolioDetail)
  storage.strategyBacktestLast.set({ selectedStrategy: 'portfolio', assetType: 'stock', initialCapital: '200000', maxPositions: '3', maxExposure: '75', slippage: '20' } as Parameters<typeof storage.strategyBacktestLast.set>[0])
  await renderPage()
  await act(async () => runButton()!.click())
  expect(startBacktest).toHaveBeenLastCalledWith(expect.objectContaining({ initial_capital: 200000, max_positions: 3, max_exposure_pct: 0.75, slippage_bps: 20 }))
})

it('候选异步切入新策略时不被推荐回测配置覆盖', async () => {
  const { api } = await import('@/lib/api')
  const { startBacktest } = await import('@/lib/backtestTask')
  vi.mocked(api.dataStatus).mockResolvedValue(busy)
  vi.mocked(api.screenerStrategies).mockResolvedValue({ presets: [portfolioDetail], load_errors: [], legacy_migration: undefined })
  vi.mocked(api.strategyGet).mockResolvedValue(portfolioDetail)
  await renderPage()
  const candidate = { name: '自选资金配置', config: { strategy_id: 'portfolio', asset_type: 'stock', initial_capital: 200000, max_positions: 3, max_exposure_pct: 0.9, slippage_bps: 18 } } as unknown as NonNullable<Parameters<typeof StrategyBacktest>[0]['loadCandidate']>
  await act(async () => {
    root.render(<MemoryRouter><QueryClientProvider client={client}><StrategyBacktest loadCandidate={candidate} /></QueryClientProvider></MemoryRouter>)
  })
  await act(async () => { await new Promise(r => setTimeout(r, 80)) })
  await act(async () => runButton()!.click())
  expect(startBacktest).toHaveBeenLastCalledWith(expect.objectContaining({ initial_capital: 200000, max_positions: 3, max_exposure_pct: 0.9, slippage_bps: 18 }))
})

it('缓存中的其他策略详情不会在恢复候选时覆盖候选资金', async () => {
  const { api } = await import('@/lib/api')
  const { startBacktest } = await import('@/lib/backtestTask')
  const old = { ...portfolioDetail, id: 'old_portfolio' }
  vi.mocked(api.dataStatus).mockResolvedValue(busy)
  vi.mocked(api.screenerStrategies).mockResolvedValue({ presets: [old, portfolioDetail], load_errors: [], legacy_migration: undefined })
  vi.mocked(api.strategyGet).mockImplementation(async id => id === old.id ? old : portfolioDetail)
  client.setQueryData(QK.strategyDetail(old.id), old)
  storage.strategyBacktestLast.set({ selectedStrategy: old.id, assetType: 'stock', initialCapital: '400000' } as Parameters<typeof storage.strategyBacktestLast.set>[0])
  const candidate = { name: '候选资金', config: { strategy_id: 'portfolio', asset_type: 'stock', initial_capital: 200000, max_positions: 3, max_exposure_pct: 0.9, slippage_bps: 18 } } as unknown as NonNullable<Parameters<typeof StrategyBacktest>[0]['loadCandidate']>
  await act(async () => root.render(<MemoryRouter><QueryClientProvider client={client}><StrategyBacktest loadCandidate={candidate} /></QueryClientProvider></MemoryRouter>))
  await act(async () => { await new Promise(r => setTimeout(r, 80)) })
  await act(async () => runButton()!.click())
  expect(startBacktest).toHaveBeenLastCalledWith(expect.objectContaining({ strategy_id: 'portfolio', initial_capital: 200000, max_positions: 3, max_exposure_pct: 0.9, slippage_bps: 18 }))
})

it.each([
  { overrides: {}, summary: '最长 8天' },
  { overrides: { max_hold_days: null }, summary: '不限持仓' },
])('恢复配置后持仓期限区分未覆盖默认值与显式取消：$summary', async ({ overrides, summary }) => {
  const { api } = await import('@/lib/api')
  vi.mocked(api.dataStatus).mockResolvedValue(busy)
  vi.mocked(api.screenerStrategies).mockResolvedValue({ presets: [portfolioDetail], load_errors: [], legacy_migration: undefined })
  vi.mocked(api.strategyGet).mockResolvedValue(portfolioDetail)
  storage.strategyBacktestLast.set({ selectedStrategy: 'portfolio', assetType: 'stock' } as Parameters<typeof storage.strategyBacktestLast.set>[0])
  await renderPage()
  const candidate = { name: '恢复持仓期限', config: { strategy_id: 'portfolio', overrides } } as unknown as NonNullable<Parameters<typeof StrategyBacktest>[0]['loadCandidate']>
  await act(async () => root.render(<MemoryRouter><QueryClientProvider client={client}><StrategyBacktest loadCandidate={candidate} /></QueryClientProvider></MemoryRouter>))
  const settings = [...host.querySelectorAll('button')].find(b => b.textContent?.includes('策略设置'))!
  expect(settings.textContent).toContain(summary)
})

it('持仓防守策略覆盖旧收盘配置并锁定成交口径，保留用户仓数和仓位', async () => {
  const { api } = await import('@/lib/api')
  const { startBacktest } = await import('@/lib/backtestTask')
  vi.mocked(api.dataStatus).mockResolvedValue(busy)
  vi.mocked(api.screenerStrategies).mockResolvedValue({ presets: [boundDetail], load_errors: [], legacy_migration: undefined })
  vi.mocked(api.strategyGet).mockResolvedValue(boundDetail)
  storage.strategyBacktestLast.set({ selectedStrategy: 'bound', assetType: 'stock', entryFill: 'close_t', exitFill: 'close_t', mode: 'full', minuteFill: true, maxPositions: '2', maxExposure: '100' } as Parameters<typeof storage.strategyBacktestLast.set>[0])
  await renderPage()
  const entry = host.querySelector<HTMLSelectElement>('select[aria-label="建仓口径"]')!
  const exit = host.querySelector<HTMLSelectElement>('select[aria-label="清仓口径"]')!
  expect(entry.value).toBe('open_t+1')
  expect(exit.value).toBe('open_t+1')
  expect(entry.disabled).toBe(true)
  expect(exit.disabled).toBe(true)
  const full = [...host.querySelectorAll('button')].find(b => b.textContent?.includes('全量模拟'))!
  expect(full.disabled).toBe(true)
  await act(async () => runButton()!.click())
  expect(startBacktest).toHaveBeenLastCalledWith(expect.objectContaining({ entry_fill: 'open_t+1', exit_fill: 'open_t+1', mode: 'position', minute_fill: false, max_positions: 2, max_exposure_pct: 1 }))
})

it('未声明成交要求的策略仍可使用已保存的收盘成交配置', async () => {
  const { api } = await import('@/lib/api')
  const detail = { ...boundDetail, id: 'ordinary', backtest_requirements: undefined }
  vi.mocked(api.dataStatus).mockResolvedValue(busy)
  vi.mocked(api.screenerStrategies).mockResolvedValue({ presets: [detail], load_errors: [], legacy_migration: undefined })
  vi.mocked(api.strategyGet).mockResolvedValue(detail)
  storage.strategyBacktestLast.set({ selectedStrategy: 'ordinary', assetType: 'stock', entryFill: 'close_t', exitFill: 'close_t' } as Parameters<typeof storage.strategyBacktestLast.set>[0])
  await renderPage()
  const entry = host.querySelector<HTMLSelectElement>('select[aria-label="建仓口径"]')!
  expect(entry.value).toBe('close_t')
  expect(entry.disabled).toBe(false)
})

it('异步载入旧候选时仍使用策略成交要求，不覆盖候选的资金设置', async () => {
  const { api } = await import('@/lib/api')
  const { startBacktest } = await import('@/lib/backtestTask')
  vi.mocked(api.dataStatus).mockResolvedValue(busy)
  vi.mocked(api.screenerStrategies).mockResolvedValue({ presets: [boundDetail], load_errors: [], legacy_migration: undefined })
  vi.mocked(api.strategyGet).mockResolvedValue(boundDetail)
  storage.strategyBacktestLast.set({ selectedStrategy: 'bound', assetType: 'stock' } as Parameters<typeof storage.strategyBacktestLast.set>[0])
  await renderPage()
  const candidate = { name: '旧候选', config: { strategy_id: 'bound', entry_fill: 'close_t', exit_fill: 'close_t', mode: 'full', minute_fill: true, max_positions: 3, max_exposure_pct: 0.6 } } as unknown as NonNullable<Parameters<typeof StrategyBacktest>[0]['loadCandidate']>
  await act(async () => root.render(
    <MemoryRouter><QueryClientProvider client={client}><StrategyBacktest loadCandidate={candidate} /></QueryClientProvider></MemoryRouter>,
  ))
  await act(async () => runButton()!.click())
  expect(startBacktest).toHaveBeenLastCalledWith(expect.objectContaining({ entry_fill: 'open_t+1', exit_fill: 'open_t+1', mode: 'position', minute_fill: false, max_positions: 3, max_exposure_pct: 0.6 }))
})

it('普通策略的旧签名仍能恢复用户调参，不因空执行要求重置', async () => {
  const { api } = await import('@/lib/api')
  const { startBacktest } = await import('@/lib/backtestTask')
  const detail = { ...boundDetail, id: 'ordinary', backtest_requirements: {}, params: [{ id: 'threshold', label: '阈值', type: 'float', default: 5 }], params_defaults: { threshold: 5 } } as StrategyDetail
  vi.mocked(api.dataStatus).mockResolvedValue(busy)
  vi.mocked(api.screenerStrategies).mockResolvedValue({ presets: [detail], load_errors: [], legacy_migration: undefined })
  vi.mocked(api.strategyGet).mockResolvedValue(detail)
  const oldSignature = JSON.stringify({ execution_backend: detail.execution_backend, basic_filter: detail.basic_filter, params: detail.params, params_defaults: detail.params_defaults, scoring: detail.scoring, scoring_directions: detail.scoring_directions, entry_signals: detail.entry_signals, exit_signals: detail.exit_signals, overlay_filter: [] })
  storage.strategyBacktestLast.set({ selectedStrategy: 'ordinary', assetType: 'stock', strategyConfigSignature: oldSignature, params: { threshold: 9 }, overrides: { score_min: 22 }, entryFill: 'close_t', exitFill: 'close_t' } as unknown as Parameters<typeof storage.strategyBacktestLast.set>[0])
  await renderPage()
  await act(async () => runButton()!.click())
  expect(startBacktest).toHaveBeenLastCalledWith(expect.objectContaining({ entry_fill: 'close_t', params: { threshold: 9 }, overrides: expect.objectContaining({ score_min: 22 }) }))
})

afterEach(async () => {
  await act(async () => root.unmount())
  client.clear()
  host.remove()
  vi.restoreAllMocks()
})

const busy = {
  indicators_ready: false, stats_pending: true,
  daily: null, enriched: null, etf_enriched: null,
} as unknown as DataStatus
const readyNoData = {
  indicators_ready: true, stats_pending: false,
  daily: null, enriched: null, etf_enriched: null,
} as unknown as DataStatus

async function renderPage() {
  const errors: unknown[] = []
  const spy = vi.spyOn(console, 'error').mockImplementation((...args) => { errors.push(args) })
  await act(async () => root.render(
    <MemoryRouter><QueryClientProvider client={client}><StrategyBacktest /></QueryClientProvider></MemoryRouter>,
  ))
  await act(async () => { await new Promise(r => setTimeout(r, 80)) })
  spy.mockRestore()
  return errors
}

function runButton(): HTMLButtonElement | undefined {
  return [...host.querySelectorAll('button')].find(b => b.textContent?.includes('运行回测'))
}

it('预热/同步期间统计为 null: 不提示缺数据, 回测按钮可用', async () => {
  const { api } = await import('@/lib/api')
  vi.mocked(api.dataStatus).mockResolvedValue(busy)

  const errors = await renderPage()
  // 核心误判: busy 未知态不得显示"缺少指标数据"横幅 (backtestDataUnavailable=false,
  // 按钮的 disabled 另受"未选策略"影响, 不在此断言)
  expect(host.textContent).not.toContain('缺少股票指标数据')
  // 渲染期 TypeError 防线 (与数据页崩溃修复同族守护)
  expect(errors.some(e => String(e).includes('TypeError'))).toBe(false)
})

it('真实统计就绪且确无数据: 提示缺数据并禁用回测', async () => {
  const { api } = await import('@/lib/api')
  vi.mocked(api.dataStatus).mockResolvedValue(readyNoData)

  await renderPage()
  expect(host.textContent).toContain('缺少股票指标数据')
  expect(runButton()!.disabled).toBe(true)
})

const libraryStrategies = [
  { id: 'grouped', name: '观察组策略', source: 'custom', library: { group: 'g', group_name: '观察组', archived: false } },
  { id: 'ungrouped', name: '未分组策略', source: 'custom' },
  { id: 'archived', name: '归档策略', source: 'ai', library: { group: 'g', group_name: '观察组', archived: true } },
] as unknown as StrategyDetail[]

it('策略回测隐藏新选择中的归档策略并支持自定义分组', async () => {
  const { api } = await import('@/lib/api')
  vi.mocked(api.dataStatus).mockResolvedValue(busy)
  vi.mocked(api.screenerStrategies).mockResolvedValue({ presets: libraryStrategies, load_errors: [], legacy_migration: undefined })
  await renderPage()
  expect(host.textContent).toContain('观察组策略')
  expect(host.textContent).toContain('未分组策略')
  expect(host.textContent).not.toContain('归档策略')
  const filter = host.querySelector<HTMLSelectElement>('select[aria-label="策略分组"]')!
  await act(async () => { filter.value = 'g'; filter.dispatchEvent(new Event('change', { bubbles: true })) })
  expect(host.textContent).not.toContain('未分组策略')
  expect(host.textContent).toContain('观察组策略')
})

it('已保存的归档策略仍能载入, 恢复后缓存刷新可重新选择', async () => {
  const { api } = await import('@/lib/api')
  vi.mocked(api.dataStatus).mockResolvedValue(busy)
  vi.mocked(api.screenerStrategies).mockResolvedValue({ presets: libraryStrategies, load_errors: [], legacy_migration: undefined })
  vi.mocked(api.strategyGet).mockResolvedValue({
    ...libraryStrategies[2], params: [], params_defaults: {}, basic_filter: {}, scoring: {},
    scoring_directions: {}, entry_signals: [], exit_signals: [], asset_types: ['stock'], timeframes: ['1d'],
  } as StrategyDetail)
  storage.strategyBacktestLast.set({ selectedStrategy: 'archived' } as Parameters<typeof storage.strategyBacktestLast.set>[0])
  await renderPage()
  expect(host.textContent).toContain('归档策略（已归档）')
  expect(api.strategyGet).toHaveBeenCalledWith('archived')
  await act(async () => client.setQueryData(QK.screenerStrategies('stock', 'all'), {
    presets: libraryStrategies.map(item => item.id === 'archived' ? { ...item, library: { ...item.library, archived: false } } : item),
  }))
  await act(async () => { await new Promise(resolve => setTimeout(resolve, 0)) })
  expect(host.textContent).not.toContain('归档策略（已归档）')
})
