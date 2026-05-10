let rendered = false;

export async function initInfoPage(container) {
  if (rendered) return;
  rendered = true;
  try {
    const data = await fetch('/examples-meta').then((r) => r.json());
    container.innerHTML = renderPage(data);
    wireToggles(container);
  } catch (err) {
    rendered = false;
    container.innerHTML = `<p class="info-error">Failed to load: ${err.message}</p>`;
  }
}

function renderPage({ examples, texts }) {
  const byEngine = (key) => examples.filter((e) => e.engine === key);
  return `
    <a href="#" class="back-btn">← back</a>

    ${tagsSection()}
    ${engineSection('Kokoro 82M', kokoroDesc(), byEngine('kokoro'), texts)}
    ${engineSection('Chatterbox', chatterboxDesc(), byEngine('chatterbox'), texts)}
    ${engineSection('Qwen3 VoiceDesign', qwen3Desc(), byEngine('qwen3'), texts)}
  `;
}

function tagsSection() {
  return `
    <section class="info-section">
      <h2>Speaker Tags</h2>
      <p>Insert <code>[name]</code> tags in your text to switch the active voice mid-generation. A tag stays active until the next one.</p>

      <h3>Chatterbox &amp; Kokoro</h3>
      <p><code>[voice_name]</code> maps to a voice file by stem — <code>[geralt]</code> uses <code>voice_samples/geralt.mp3</code>, <code>[af_heart]</code> uses the Kokoro voice of that name. Falls back to the selected voice if the name isn't found. The voice description in the tag (if any) is ignored for these engines.</p>

      <h3>Qwen3</h3>
      <p>Qwen3 uses a two-step workflow. The first time a name appears, include a voice description:</p>
      <pre class="code-block">[narrator: Calm professional male narrator voice]
Text here...</pre>
      <p>The VoiceDesign model generates a ~15 s reference sample and saves it to <code>voice_samples/qwen3/narrator.wav</code>. All subsequent uses of <code>[narrator]</code> — in the same or any future generation — load that saved sample and use voice cloning instead of generating a new one.</p>
      <p>Use <code>[name: description]</code> again at any point to regenerate the sample (the saved file is overwritten). If <code>[name]</code> is used without a description and no sample exists yet, it falls back to the default voice description field.</p>
      <p>Saved Qwen3 voices appear in the autocomplete when you type <code>[</code> in the text field.</p>
    </section>
  `;
}

function kokoroDesc() {
  return `<p>Smallest and by far the fastest model. Works using pre-trained voice models. Mediocre results, especially when mixing multiple voices.</p>`;
}

function chatterboxDesc() {
  return `
    <p>Large model, rather slow compared to Kokoro. Works by cloning voices from short audio samples. Very solid results.</p>
    <div class="warn">Chatterbox is very picky about input length. This applies to individual voice segments too: switching voices mid-text creates short utterances that fall outside the model's comfortable range. Long segments are split automatically, but short ones (like the "Brother Avien" line in the with-tags example) can sound off. There's no workaround for very short utterances.</div>
  `;
}

function qwen3Desc() {
  return `<p>Large model, <strong>very slow</strong>. Works using a text description of the speaker voice — no audio sample needed. Very solid results. See the Speaker Tags section above for the VoiceDesign → cloning workflow.</p>`;
}

function engineSection(title, descHtml, examples, texts) {
  if (!examples.length) return '';
  return `
    <section class="info-section">
      <h2>${title}</h2>
      <div class="engine-desc">${descHtml}</div>
      <div class="examples-list">
        ${examples.map((ex) => exampleCard(ex, texts)).join('')}
      </div>
    </section>
  `;
}

function exampleCard(ex, texts) {
  const text = esc(texts[ex.text_file] || '');
  const open = ex.with_tags;
  return `
    <div class="example-card">
      <div class="example-label">${esc(ex.label)}</div>
      <audio controls src="/examples/${ex.filename}"></audio>
      <button class="text-toggle">${open ? 'hide text ▲' : 'show text ▼'}</button>
      <pre class="example-text" style="display: ${open ? 'block' : 'none'}">${text}</pre>
    </div>
  `;
}

function wireToggles(container) {
  container.querySelectorAll('.text-toggle').forEach((btn) => {
    btn.addEventListener('click', () => {
      const pre = btn.nextElementSibling;
      const hidden = pre.style.display === 'none';
      pre.style.display = hidden ? 'block' : 'none';
      btn.textContent = hidden ? 'hide text ▲' : 'show text ▼';
    });
  });
}

function esc(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
