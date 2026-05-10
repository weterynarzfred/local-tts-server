import Fuse from 'fuse.js'

let fuse = null
let voiceList = []
let activeIndex = -1
let textarea = null

const dropdown = document.createElement('div')
dropdown.id = 'autocomplete'
document.body.appendChild(dropdown)

export function initAutocomplete(el) {
  textarea = el
  textarea.addEventListener('input', handleInput)
  textarea.addEventListener('keydown', handleKeydown)
  textarea.addEventListener('click', () => { if (getQuery() === null) hide() })
  document.addEventListener('selectionchange', () => {
    if (document.activeElement === textarea && getQuery() === null) hide()
  })
  document.addEventListener('mousedown', (e) => {
    if (!dropdown.contains(e.target) && e.target !== textarea) hide()
  })
  window.addEventListener('resize', () => {
    if (dropdown.style.display !== 'none') positionDropdown()
  })
}

export function updateVoices(voices) {
  voiceList = voices.map((v) => ({
    stem: v.filename.replace(/\.[^.]+$/, ''),
    label: v.label,
  }))
  fuse =
    voiceList.length > 0
      ? new Fuse(voiceList, { keys: ['stem', 'label'], threshold: 0.4, minMatchCharLength: 1 })
      : null
  if (dropdown.style.display !== 'none') handleInput()
}

function getQuery() {
  const pos = textarea.selectionStart
  const before = textarea.value.substring(0, pos)
  const open = before.lastIndexOf('[')
  if (open === -1) return null
  const between = before.substring(open + 1)
  if (between.includes(']') || between.includes(':')) return null
  return between
}

function handleInput() {
  const query = getQuery()
  if (query === null) { hide(); return }

  const results =
    query === '' || !fuse ? voiceList : fuse.search(query).map((r) => r.item)

  if (results.length === 0) { hide(); return }
  show(results)
}

function show(results) {
  activeIndex = -1
  dropdown.innerHTML = ''

  for (const voice of results) {
    const item = document.createElement('div')
    item.className = 'ac-item'
    item.dataset.stem = voice.stem
    item.innerHTML =
      `<span class="ac-stem">[${voice.stem}]</span>` +
      `<span class="ac-label">${voice.label}</span>`
    item.addEventListener('mousedown', (e) => {
      e.preventDefault()
      insert(voice.stem)
    })
    dropdown.appendChild(item)
  }

  positionDropdown()
  dropdown.style.display = 'block'
}

function hide() {
  dropdown.style.display = 'none'
  activeIndex = -1
}

function positionDropdown() {
  const rect = textarea.getBoundingClientRect()
  dropdown.style.left = `${rect.left}px`
  dropdown.style.top = `${rect.bottom + 4}px`
  dropdown.style.width = `${rect.width}px`
}

function setActive(index) {
  const items = dropdown.querySelectorAll('.ac-item')
  if (!items.length) return
  activeIndex = Math.max(0, Math.min(index, items.length - 1))
  items.forEach((item, i) => item.classList.toggle('active', i === activeIndex))
  items[activeIndex]?.scrollIntoView({ block: 'nearest' })
}

function handleKeydown(e) {
  if (dropdown.style.display === 'none') return
  const items = dropdown.querySelectorAll('.ac-item')

  if (e.key === 'ArrowDown') {
    e.preventDefault()
    setActive(activeIndex + 1)
  } else if (e.key === 'ArrowUp') {
    e.preventDefault()
    setActive(activeIndex === -1 ? items.length - 1 : activeIndex - 1)
  } else if (e.key === 'Enter' && activeIndex >= 0) {
    e.preventDefault()
    insert(items[activeIndex].dataset.stem)
  } else if (e.key === 'Tab') {
    e.preventDefault()
    const i = activeIndex === -1 ? 0 : activeIndex
    if (items[i]) insert(items[i].dataset.stem)
  } else if (e.key === 'Escape') {
    hide()
  }
}

function insert(stem) {
  if (!stem) return
  const pos = textarea.selectionStart
  const text = textarea.value
  const before = text.substring(0, pos)
  const open = before.lastIndexOf('[')

  textarea.value = text.substring(0, open) + '[' + stem + ']' + text.substring(pos)
  const newPos = open + stem.length + 2
  textarea.setSelectionRange(newPos, newPos)
  textarea.focus()
  hide()
}
