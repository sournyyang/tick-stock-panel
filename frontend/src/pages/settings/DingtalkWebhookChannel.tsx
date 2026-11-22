import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ChevronDown } from 'lucide-react'
import { api } from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import { toast } from '@/components/Toast'

export function DingtalkWebhookChannel({ url, secretSet, selected, onToggle }: {
  url: string
  secretSet: boolean
  selected: boolean
  onToggle: (selected: boolean) => void
}) {
  const qc = useQueryClient()
  const [open, setOpen] = useState(false)
  const [draft, setDraft] = useState(url)
  const [secretDraft, setSecretDraft] = useState('')
  const config = useQuery({
    queryKey: QK.dingtalkWebhook,
    queryFn: api.getDingtalkWebhook,
    enabled: open,
    gcTime: 0,
    retry: false,
    refetchOnWindowFocus: false,
  })
  useEffect(() => {
    setDraft(url)
  }, [url])
  useEffect(() => {
    if (config.data) setSecretDraft(config.data.secret)
  }, [config.data])
  const test = useMutation({ mutationFn: () => api.sendTestWebhook('dingtalk') })
  const save = useMutation({
    mutationFn: (config: { url: string; secret?: string }) => api.updateDingtalkWebhook(config.url, config.secret),
    onMutate: () => test.reset(),
    onSuccess: (data, saved) => {
      const secret = data.dingtalk_webhook_url ? saved.secret ?? config.data?.secret ?? '' : ''
      setDraft(data.dingtalk_webhook_url)
      setSecretDraft(secret)
      qc.setQueryData(QK.dingtalkWebhook, { url: data.dingtalk_webhook_url, secret })
      toast('钉钉 Webhook 已保存', 'success')
      qc.invalidateQueries({ queryKey: QK.preferences })
    },
  })
  const secretChanged = secretDraft.trim() !== (config.data?.secret ?? '')
  const changed = draft.trim() !== url || secretChanged
  const busy = save.isPending || test.isPending
  const configUnavailable = config.isPending || config.isError
  const inputClass = 'h-9 w-full rounded-btn border border-border bg-base px-3 text-xs font-mono text-foreground focus:outline-none focus:border-accent/50'
  const buttonClass = 'px-3 py-1.5 rounded-btn bg-accent text-white text-xs font-medium disabled:opacity-50 cursor-pointer hover:bg-accent/90'

  return (
    <div className="rounded-btn border border-border/60 bg-base/40 overflow-hidden">
      <div className="flex items-center gap-2 px-2.5 py-2">
        <input
          type="checkbox" checked={selected} onChange={e => onToggle(e.target.checked)}
          aria-label="钉钉作为新建规则的默认推送渠道" className="h-3 w-3 accent-accent cursor-pointer"
        />
        <button type="button" onClick={() => setOpen(value => !value)} aria-expanded={open}
          className="flex flex-1 min-w-0 items-center gap-2 text-left cursor-pointer">
          <span className="text-[11px] font-medium text-foreground">钉钉</span>
          <span className="text-[9px] text-muted">群机器人 Webhook</span>
          {selected && <span className="rounded bg-accent/15 px-1 py-px text-[9px] text-accent">默认</span>}
          <span className={`ml-auto shrink-0 text-[9px] ${url ? 'text-emerald-500' : 'text-warning'}`}>
            {url ? '已配置' : '未配置'}
          </span>
          <ChevronDown className={`h-3 w-3 shrink-0 text-muted transition-transform ${open ? 'rotate-180' : ''}`} />
        </button>
      </div>
      {open && (
        <div className="border-t border-border/60 bg-base/30 p-3">
          <label className="block space-y-1.5">
            <span className="text-[11px] text-muted">钉钉 Webhook 地址</span>
            <input type="text" value={draft} disabled={busy || configUnavailable} autoComplete="off" spellCheck={false}
              onChange={e => { setDraft(e.target.value); save.reset(); test.reset() }}
              placeholder="https://oapi.dingtalk.com/robot/send?access_token=..." className={inputClass} />
          </label>
          <label className="block mt-2 space-y-1.5">
            <span className="text-[11px] text-muted">加签密钥（可选）</span>
            <input type="text" value={secretDraft} disabled={busy || configUnavailable} autoComplete="off" spellCheck={false}
              onChange={e => { setSecretDraft(e.target.value); save.reset(); test.reset() }}
              placeholder="机器人启用加签时填写 SEC..." className={inputClass} />
          </label>
          {config.isPending && <p role="status" className="mt-2 text-[11px] text-muted">正在读取已保存配置…</p>}
          {config.isError && <p role="alert" className="mt-2 text-[11px] text-danger">
            {config.error.message || '读取配置失败'}
            <button type="button" onClick={() => { void config.refetch() }} className="ml-2 underline">重试读取配置</button>
          </p>}
          {save.isError && <p role="alert" className="mt-2 text-[11px] text-danger">{String(save.error?.message ?? '保存失败')}</p>}
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <button type="button" disabled={busy || configUnavailable || !changed} className={buttonClass}
              onClick={() => save.mutate({ url: draft.trim(), ...(secretChanged ? { secret: secretDraft.trim() } : {}) })}>
              {save.isPending ? '保存中…' : '保存'}
            </button>
            <button type="button" disabled={!url || changed || busy || configUnavailable}
              onClick={() => test.mutate()} className={buttonClass}>
              {test.isPending ? '发送中…' : '发送测试消息'}
            </button>
            {(config.data ? !!config.data.secret : secretSet) && <button type="button" disabled={busy || configUnavailable || changed}
              onClick={() => save.mutate({ url, secret: '' })} className="text-[11px] text-muted hover:text-danger disabled:opacity-50">
              清除加签密钥
            </button>}
            {url && <button type="button" disabled={busy || configUnavailable}
              onClick={() => save.mutate({ url: '' })} className="text-[11px] text-muted hover:text-danger disabled:opacity-50">
              清空配置
            </button>}
          </div>
          {test.data && <p role="status" className={`mt-2 text-[11px] ${test.data.ok ? 'text-emerald-500' : 'text-danger'}`}>{test.data.detail}</p>}
          {test.isError && <p role="alert" className="mt-2 text-[11px] text-danger">{String(test.error?.message ?? '测试失败')}</p>}
          <p className="mt-3 text-[10px] text-muted leading-relaxed">
            在钉钉群添加自定义机器人，复制原始 Webhook 地址；启用加签时填写密钥。建议使用加签安全设置。
            修改后先保存，再发送测试消息；成功后请到群内确认收件。
            <a href="https://open.dingtalk.com/document/robots/customize-robot-security-settings" target="_blank" rel="noreferrer" className="ml-1 text-accent hover:underline">接入说明 →</a>
          </p>
        </div>
      )}
    </div>
  )
}
