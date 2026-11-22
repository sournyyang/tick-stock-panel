"""三倍量整理反弹：成交规模排序、MA3及信号绑定收盘防守。"""
import numpy as np

from app.backtest.matrix import make_signal_matrix

META = {
    "id": "custom_triple_volume_ma3_defense",
    "name": "三倍量整理反弹·单仓30%",
    "version": "1.0.1",
    "description": "启动三倍量穿三线、次日缩量，启动后第7～12根有效日K中阳反弹；取消次日新高、整理守MA20和上穿启动收盘要求。成交规模优先，MA3或MAX(启动低点,信号收盘×0.97)收盘失守后次日开盘退出。组合回测配置：最多1仓、30%买入预算、单边12bp。",
    "tags": ["三倍量", "整理反弹", "收盘防守"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "default_entry_fill": "open_t+1",
    "default_exit_fill": "open_t+1",
    "backtest_requirements": {
        "entry_fill": "open_t+1",
        "exit_fill": "open_t+1",
        "mode": "position",
        "minute_fill": False,
    },
    "params": [],
    "scoring": {},
    "order_by": "score",
    "descending": True,
    "limit": 100,
}
BASIC_FILTER = {
    "price_min": 0,
    "price_max": None,
    "market_cap_min": None,
    "float_cap_min": None,
    "float_cap_max": None,
    "amount_min": None,
    "amount_max": None,
    "turnover_min": None,
    "turnover_max": None,
    "exclude_st": True,
    "exclude_new_days": 0,
    "boards": ["沪主板", "深主板", "创业板", "科创板"],
}
EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["signal_triple_volume_breakout"]
EXIT_SIGNALS = ["signal_close_below_ma3"]
STOP_LOSS = None
MAX_HOLD_DAYS = None


def _mean(values,window):
    result=np.full(len(values),np.nan)
    sums=np.concatenate(([0.],np.cumsum(values,dtype=np.float64)))
    result[window-1:]=(sums[window:]-sums[:-window])/window
    return result


def _economic_rank(times,values):
    rank=np.zeros(len(times),dtype=np.float32)
    allowed=np.ones(len(times),dtype=bool)
    for t in np.unique(times):
        ids=np.flatnonzero(times==t)
        data=values[ids]
        # Primary: log(raw close * volume); secondary: twelve economic fields.
        # Identical complete vectors are rejected together, with no symbol fallback.
        economic=np.column_stack((data[:,8],data))
        order=np.lexsort(tuple(economic[:,j] for j in range(economic.shape[1]-1,-1,-1)))
        same=np.all(economic[order][1:]==economic[order][:-1],axis=1)
        ambiguous=np.zeros(len(order),dtype=bool)
        ambiguous[1:] |= same
        ambiguous[:-1] |= same
        allowed[ids[order[ambiguous]]]=False
        kept=ids[order[~ambiguous]]
        rank[kept]=np.arange(1,len(kept)+1,dtype=np.float32)
    return rank,allowed


class TripleVolumeDefense:
    def required_fields(self):
        return frozenset({"open","high","low","close","volume","raw_close","price_limit_pct"})

    def required_warmup_bars(self,params):
        # The setup itself needs 20 bars plus at most twelve setup/wash bars.
        return 60

    def compute_signals(self,market,params):
        entry=np.zeros(market.shape,dtype=np.uint8)
        exit_=np.zeros(market.shape,dtype=np.uint8)
        floors=np.full(market.shape,np.nan,dtype=np.float32)
        score=np.zeros(market.shape,dtype=np.float32)
        event_times=[]
        event_assets=[]
        economic=[]
        raw=market.field("raw_close")
        for a in range(market.shape[1]):
            valid=(np.isfinite(market.open[:,a])&np.isfinite(market.high[:,a])&np.isfinite(market.low[:,a])&np.isfinite(market.close[:,a])&np.isfinite(market.volume[:,a])&(market.open[:,a]>0)&(market.close[:,a]>0)&(market.volume[:,a]>0))
            ids=np.flatnonzero(valid)
            if not len(ids):
                continue
            o,h,low,c,v=(x[ids,a].astype(np.float64) for x in [market.open,market.high,market.low,market.close,market.volume])
            ma3,ma5,ma10,ma20=(_mean(c,n) for n in [3,5,10,20])
            exit_[ids,a]=c<ma3
            if len(ids)<27:
                continue
            key=(c>=o*1.05)&(h>=np.maximum.reduce([ma5,ma10,ma20]))&(low<=np.minimum.reduce([ma5,ma10,ma20]))
            key[0]=False
            key[1:] &= v[1:]>=v[:-1]*3
            lock=np.zeros(len(ids),dtype=bool)
            lock[1:]=key[:-1]&(v[1:]<v[:-1])
            indices=np.arange(len(ids))
            last_lock=np.maximum.accumulate(np.where(lock,indices,-1))
            position=np.zeros(len(ids),dtype=bool)
            locks=np.flatnonzero(lock)
            for j,k in enumerate(locks):
                end=int(locks[j+1]) if j+1<len(locks) else len(ids)
                if end<=k+1:
                    continue
                wash=np.maximum.accumulate(h[k+1:end])
                age=indices[k+1:end]-k
                position[k+1:end]=(age>=5)&(age<=10)&(wash<=max(h[k-1],h[k]))
            prior_position=np.concatenate(([False],position[:-1]))
            candidates=prior_position&(last_lock>=1)&(c>=o*1.04)
            candidates[1:] &= v[1:]>=v[:-1]*1.5
            for i in np.flatnonzero(candidates):
                anchor=int(last_lock[i])-1
                if i<20 or v[i]>=v[anchor]:
                    continue
                t=int(ids[i])
                bar_range=h[i]-low[i]
                if bar_range<=0 or h[i]+1e-5<max(o[i],c[i]) or low[i]-1e-5>min(o[i],c[i]) or not np.isfinite(raw[t,a]) or raw[t,a]<=0:
                    continue
                tr=np.maximum.reduce([h[i-14:i]-low[i-14:i],np.abs(h[i-14:i]-c[i-15:i-1]),np.abs(low[i-14:i]-c[i-15:i-1])])
                atr=np.mean(tr)
                if atr<=0 or np.min(low[i-5:i])<=0:
                    continue
                returns=c[i-19:i+1]/c[i-20:i]-1
                values=[c[i]/o[i]-1,(c[i]-low[i])/bar_range,c[i]/c[anchor]-1,-v[i]/v[anchor],v[i]/v[i-1],c[i]/c[i-20]-1,-np.std(returns,ddof=1),-(np.max(h[i-5:i])/np.min(low[i-5:i])-1),np.log(float(raw[t,a])*v[i]),np.mean(c[i-4:i+1])/np.mean(c[i-9:i-4])-1,(c[i]-o[i])/bar_range,(c[i]-o[i])/atr]
                floor=max(low[anchor],c[i]*.97)
                if not np.isfinite(values).all() or not np.isfinite(floor) or floor<=0:
                    continue
                event_times.append(t)
                event_assets.append(a)
                economic.append(values)
                floors[t,a]=floor
        if event_times:
            times=np.asarray(event_times)
            assets=np.asarray(event_assets)
            rank,allowed=_economic_rank(times,np.asarray(economic))
            entry[times[allowed],assets[allowed]]=1
            score[times[allowed],assets[allowed]]=rank[allowed]
        return make_signal_matrix(
            market.shape,entry=entry,exit=exit_,score=score,
            entry_signal_code=np.where(entry,0,-1).astype(np.int16),
            exit_signal_code=np.where(exit_,0,-1).astype(np.int16),
            entry_signal_ids=tuple(ENTRY_SIGNALS),exit_signal_ids=tuple(EXIT_SIGNALS),
            entry_close_floor=floors,
        )


MATRIX_STRATEGY = TripleVolumeDefense()
