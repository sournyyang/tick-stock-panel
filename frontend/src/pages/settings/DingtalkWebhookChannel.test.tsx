// @vitest-environment jsdom
import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { DingtalkWebhookChannel } from './DingtalkWebhookChannel'

const apiMock = vi.hoisted(() => ({ getDingtalkWebhook: vi.fn(), updateDingtalkWebhook: vi.fn(), sendTestWebhook: vi.fn() }))
vi.mock('@/lib/api', () => ({ api: apiMock }))
vi.mock('@/components/Toast', () => ({ toast: vi.fn() }))

let root: Root | null = null
let client: QueryClient
const container = document.createElement('div')
document.body.appendChild(container)
const savedUrl = 'https://oapi.dingtalk.com/robot/send?access_token=test'
beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  vi.resetAllMocks()
  apiMock.getDingtalkWebhook.mockResolvedValue({ url: savedUrl, secret: 'SEC-saved' })
  apiMock.updateDingtalkWebhook.mockImplementation(async (url, secret) => ({ dingtalk_webhook_url: url, dingtalk_webhook_secret_set: secret !== '' && !!url }))
  apiMock.sendTestWebhook.mockResolvedValue({ ok: true, detail: '测试消息已发送，请检查钉钉群' })
  client = new QueryClient({ defaultOptions: { mutations: { retry: false } } })
})
afterEach(() => {
  act(() => { root?.unmount() })
  root = null
  client.clear()
  container.innerHTML = ''
})

async function render(url = savedUrl, secretSet = true, onToggle = vi.fn()) {
  if (!secretSet) apiMock.getDingtalkWebhook.mockResolvedValue({ url, secret: '' })
  root = createRoot(container)
  act(() => { root!.render(
    <QueryClientProvider client={client}>
      <DingtalkWebhookChannel url={url} secretSet={secretSet} selected={false} onToggle={onToggle} />
    </QueryClientProvider>,
  ) })
  act(() => { container.querySelector<HTMLButtonElement>('button[aria-expanded]')!.click() })
  await act(async () => {
    await new Promise(resolve => setTimeout(resolve, 10))
  })
}

function button(label: string) {
  return [...container.querySelectorAll<HTMLButtonElement>('button')].find(b => b.textContent === label)!
}

async function click(label: string) {
  await act(async () => {
    button(label).click()
    await new Promise(resolve => setTimeout(resolve, 10))
  })
}

function input(index: number, value: string) {
  act(() => {
    const element = container.querySelectorAll<HTMLInputElement>('input[type=text]')[index]
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(element, value)
    element.dispatchEvent(new Event('input', { bubbles: true }))
  })
}

it('unconfigured channel disables testing and allows selecting the rule default', async () => {
  const toggle = vi.fn()
  await render('', false, toggle)
  expect(container.textContent).toContain('未配置')
  expect(button('发送测试消息').disabled).toBe(true)
  act(() => { container.querySelector<HTMLInputElement>('input[type=checkbox]')!.click() })
  expect(toggle).toHaveBeenCalledWith(true)
  expect(apiMock.sendTestWebhook).not.toHaveBeenCalled()
})

it('changing the URL disables test until saved and preserves the secret by omission', async () => {
  await render()
  expect(container.querySelectorAll<HTMLInputElement>('input[type=text]')[1].value).toBe('SEC-saved')
  input(0, savedUrl + '-new')
  expect(button('发送测试消息').disabled).toBe(true)
  await click('保存')
  expect(apiMock.updateDingtalkWebhook).toHaveBeenCalledWith(savedUrl + '-new', undefined)
  expect(apiMock.sendTestWebhook).not.toHaveBeenCalled()
})

it('shows saved URL and signing secret in plaintext and retains a new secret after save', async () => {
  await render()
  expect(container.querySelectorAll<HTMLInputElement>('input[type=text]')[0].value).toBe(savedUrl)
  expect(container.querySelectorAll<HTMLInputElement>('input[type=text]')[1].value).toBe('SEC-saved')
  input(1, 'SEC-new')
  await click('保存')
  expect(apiMock.updateDingtalkWebhook).toHaveBeenCalledWith(savedUrl, 'SEC-new')
  expect(container.querySelectorAll<HTMLInputElement>('input[type=text]')[1].value).toBe('SEC-new')
})

it('explicitly clearing a secret sends an empty string and clearing the channel sends an empty URL', async () => {
  await render()
  await click('清除加签密钥')
  expect(apiMock.updateDingtalkWebhook).toHaveBeenLastCalledWith(savedUrl, '')
  await click('清空配置')
  expect(apiMock.updateDingtalkWebhook).toHaveBeenLastCalledWith('', undefined)
})

it('shows business rejection as failure and prompts checking the recipient on success', async () => {
  await render()
  apiMock.sendTestWebhook.mockResolvedValueOnce({ ok: false, detail: '钉钉推送失败：请检查加签密钥' })
  await click('发送测试消息')
  expect(apiMock.sendTestWebhook).toHaveBeenCalledWith('dingtalk')
  expect(container.querySelector('[role=status]')?.textContent).toContain('推送失败')
  await click('发送测试消息')
  expect(container.querySelector('[role=status]')?.textContent).toContain('请检查钉钉群')
})

it('shows save errors without clearing the secret draft', async () => {
  await render()
  apiMock.updateDingtalkWebhook.mockRejectedValueOnce(new Error('地址非法'))
  input(1, 'SEC-new')
  await click('保存')
  expect(container.querySelector('[role=alert]')?.textContent).toContain('地址非法')
  expect(container.querySelectorAll<HTMLInputElement>('input[type=text]')[1].value).toBe('SEC-new')
})


it('allows deleting the visible secret and blocks testing while unsaved', async () => {
  await render()
  input(1, '')
  expect(button('发送测试消息').disabled).toBe(true)
  await click('保存')
  expect(apiMock.updateDingtalkWebhook).toHaveBeenCalledWith(savedUrl, '')
  expect(container.querySelectorAll<HTMLInputElement>('input[type=text]')[1].value).toBe('')
  expect(button('发送测试消息').disabled).toBe(false)
})

it('blocks editing when the saved secret cannot be loaded and allows retrying', async () => {
  apiMock.getDingtalkWebhook.mockRejectedValueOnce(new Error('配置读取失败'))
  await render()
  expect(container.querySelector('[role=alert]')?.textContent).toContain('配置读取失败')
  expect(button('保存').disabled).toBe(true)
  await click('重试读取配置')
  expect(container.querySelectorAll<HTMLInputElement>('input[type=text]')[1].value).toBe('SEC-saved')
})
