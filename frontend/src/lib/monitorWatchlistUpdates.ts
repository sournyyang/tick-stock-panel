import type { QueryClient } from '@tanstack/react-query'
import { WATCHLIST_DATA_QUERY_PREFIXES } from './queryKeys'

/** 后台写入、重连及回到页面时共用刷新出口, 不受行情刷新开关限制。 */
export function registerMonitorWatchlistUpdates(stream: EventSource, client: QueryClient) {
  const refresh = () => {
    void client.invalidateQueries({
      predicate: query => WATCHLIST_DATA_QUERY_PREFIXES.has(String(query.queryKey[0])),
      refetchType: document.visibilityState === 'hidden' ? 'none' : 'active',
    })
  }
  const onVisible = () => { if (document.visibilityState !== 'hidden') refresh() }
  const dispose = () => {
    stream.removeEventListener('watchlist_updated', refresh)
    document.removeEventListener('visibilitychange', onVisible)
  }
  stream.addEventListener('watchlist_updated', refresh)
  document.addEventListener('visibilitychange', onVisible)
  return { refresh, dispose }
}
