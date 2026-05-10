import './style.css'
import { initAutocomplete, updateVoices } from './autocomplete.js'
import { initInfoPage } from './info.js'

const form = document.getElementById('form')
const engineSelect = document.getElementById('engine')
const submitBtn = document.getElementById('submit-btn')
const statusEl = document.getElementById('status')
const result = document.getElementById('result')
const player = document.getElementById('player')
const downloadLink = document.getElementById('result-download')
const errorBox = document.getElementById('error')
const progressBar = document.getElementById('progress-bar')
const progress = document.getElementById('progress')
const voiceSelect = document.getElementById('voice')
const voiceLabel = document.getElementById('voice-label')
const voiceLabelText = document.getElementById('voice-label-text')
const exaggerationLabel = document.getElementById('exaggeration-label')
const speedLabel = document.getElementById('speed-label')
const voiceDescriptionLabel = document.getElementById('voice-description-label')
const languageLabel = document.getElementById('language-label')
const textarea = document.getElementById('text')

const mainPage = document.getElementById('page-main')
const infoPage = document.getElementById('page-info')

function syncEngineFields() {
  const engine = engineSelect.value
  exaggerationLabel.style.display = engine === 'chatterbox' ? '' : 'none'
  speedLabel.style.display = engine === 'kokoro' ? '' : 'none'
  voiceLabel.style.display = engine === 'qwen3' ? 'none' : ''
  voiceLabelText.textContent = engine === 'chatterbox' ? 'voice sample' : 'voice'
  voiceDescriptionLabel.style.display = engine === 'qwen3' ? '' : 'none'
  languageLabel.style.display = engine === 'qwen3' ? '' : 'none'
}

function showPage(hash) {
  if (hash === '#info') {
    mainPage.style.display = 'none'
    infoPage.style.display = ''
    initInfoPage(infoPage)
  } else {
    mainPage.style.display = ''
    infoPage.style.display = 'none'
    syncEngineFields()
  }
}
window.addEventListener('hashchange', () => showPage(window.location.hash))
showPage(window.location.hash)

let allVoices = { chatterbox: [], kokoro: [], qwen3: [] }

initAutocomplete(textarea)

;(async () => {
  try {
    allVoices = await fetch('/voices').then((r) => r.json())
    const engine = engineSelect.value
    if (engine !== 'qwen3') populateVoices(engine)
    updateVoices(allVoices[engine] || [])
  } catch {}
})()

function populateVoices(engine) {
  const voices = allVoices[engine] || []
  voiceSelect.innerHTML = engine === 'chatterbox' ? "<option value=''>– none –</option>" : ''
  for (const v of voices) {
    const opt = document.createElement('option')
    opt.value = v.filename
    opt.textContent = v.label
    voiceSelect.appendChild(opt)
  }
}

engineSelect.addEventListener('change', () => {
  syncEngineFields()
  const engine = engineSelect.value
  if (engine !== 'qwen3') populateVoices(engine)
  updateVoices(allVoices[engine] || [])
})

form.addEventListener('submit', async (e) => {
  e.preventDefault()

  submitBtn.disabled = true
  statusEl.textContent = 'Generating...'
  result.hidden = true
  errorBox.hidden = true
  const showProgress = engineSelect.value !== 'qwen3'
  progress.value = 0
  progressBar.hidden = !showProgress

  try {
    const res = await fetch('/synthesize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams(new FormData(form)),
    })

    if (!res.ok) {
      const text = await res.text()
      throw new Error(text || `Server error ${res.status}`)
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    outer: while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop()

      let eventType = null
      let eventData = null
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim()
        } else if (line.startsWith('data: ')) {
          eventData = line.slice(6).trim()
        } else if (line === '' && eventType && eventData) {
          const data = JSON.parse(eventData)
          if (eventType === 'progress') {
            progress.value = data.value
          } else if (eventType === 'done') {
            progress.value = 1
            player.src = data.url
            downloadLink.href = data.url
            downloadLink.textContent = data.filename
            downloadLink.download = data.filename
            result.hidden = false
            result.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
            player.play()
            break outer
          } else if (eventType === 'error') {
            throw new Error(data.error)
          }
          eventType = null
          eventData = null
        }
      }
    }
  } catch (err) {
    errorBox.textContent = err.message
    errorBox.hidden = false
  } finally {
    submitBtn.disabled = false
    statusEl.textContent = ''
    progressBar.hidden = true
  }
})
