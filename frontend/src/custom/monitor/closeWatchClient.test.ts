// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest'
import { ApiError } from '@/lib/api'
import { closeWatchApi } from './closeWatchClient'

afterEach(() => vi.unstubAllGlobals())
it('uses the existing API origin and preserves authentication errors', async () => {
  const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: '认证失败' }), { status: 401 }))
  vi.stubGlobal('fetch', fetcher)
  await expect(closeWatchApi.state()).rejects.toBeInstanceOf(ApiError)
  await expect(closeWatchApi.state()).rejects.toThrow('未登录 (401)')
  expect(fetcher.mock.calls[0][0]).toBe('/api/custom/close-watch/state')
})
it('sends only monitoring configuration and keeps validation errors readable', async () => {
  const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: [{ msg: '尾盘开始时间不合法' }] }), { status: 422 }))
  vi.stubGlobal('fetch', fetcher)
  await expect(closeWatchApi.configure('demo', true, '14:50')).rejects.toThrow('尾盘开始时间不合法')
  expect(JSON.parse(fetcher.mock.calls[0][1].body)).toEqual({ strategy_id: 'demo', enabled: true, start_time: '14:50' })
})
