// @vitest-environment jsdom
import { QueryClient } from '@tanstack/react-query'
import { expect, it } from 'vitest'
import { QK } from './queryKeys'
import { registerMonitorWatchlistUpdates } from './monitorWatchlistUpdates'

it('refreshes members, groups and warnings in all windows without touching financials', () => {
  const clients = [new QueryClient(), new QueryClient()]
  const streams = [new EventTarget(), new EventTarget()]
  const keys = [QK.watchlist, QK.watchlistGroups, QK.watchlistEnriched('ext'), QK.monitorRules]
  const bindings = clients.map((client, i) => {
    for (const key of [...keys, ['financials']]) client.setQueryData(key, {})
    return registerMonitorWatchlistUpdates(streams[i] as EventSource, client)
  })
  streams.forEach(stream => stream.dispatchEvent(new Event('watchlist_updated')))
  for (const client of clients) {
    keys.forEach(key => expect(client.getQueryState(key)?.isInvalidated).toBe(true))
    expect(client.getQueryState(['financials'])?.isInvalidated).toBe(false)
  }
  bindings.forEach(binding => binding.dispose())
  clients.forEach(client => client.clear())
})

it('recovers missed notifications on visibility and reconnect and cleans up listeners', () => {
  const client = new QueryClient()
  const stream = new EventTarget()
  const binding = registerMonitorWatchlistUpdates(stream as EventSource, client)
  client.setQueryData(QK.watchlist, {})
  binding.refresh()
  expect(client.getQueryState(QK.watchlist)?.isInvalidated).toBe(true)
  client.setQueryData(QK.watchlist, {})
  document.dispatchEvent(new Event('visibilitychange'))
  expect(client.getQueryState(QK.watchlist)?.isInvalidated).toBe(true)
  client.setQueryData(QK.watchlist, {})
  binding.dispose()
  stream.dispatchEvent(new Event('watchlist_updated'))
  document.dispatchEvent(new Event('visibilitychange'))
  expect(client.getQueryState(QK.watchlist)?.isInvalidated).toBe(false)
  client.clear()
})
