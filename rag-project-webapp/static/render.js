// Build a deliberately small Markdown subset with DOM nodes, never innerHTML.
// Raw HTML, links, images, and unsupported syntax remain inert text.
function inline(parent, text, citationTargets) {
  const tokens = /(\[search\d+:\d+\]|\*\*[^*\n]+\*\*|__[^_\n]+__|`[^`\n]+`|\*[^*\n]+\*)/g;
  let offset = 0;
  for (const match of text.matchAll(tokens)) {
    parent.append(document.createTextNode(text.slice(offset, match.index)));
    const token = match[0];
    if (token.startsWith('[search')) {
      const label = token.slice(1, -1);
      const target = citationTargets.get(label);
      if (target) {
        const link = document.createElement('a');
        link.className = 'citation'; link.href = `#${target.details.id}`;
        link.textContent = `[${target.number}]`;
        link.setAttribute('aria-label', `Source ${target.number}: ${target.title}`);
        link.title = `${target.title} · Ragas paper`;
        link.addEventListener('click', event => {
          event.preventDefault();
          target.details.open = true;
          target.details.classList.remove('source-highlight');
          void target.details.offsetWidth;
          target.details.classList.add('source-highlight');
          target.summary.focus({preventScroll:true});
          target.details.scrollIntoView({behavior:'smooth', block:'center'});
        });
        parent.append(link);
      } else {
        const missing = document.createElement('span');
        missing.className = 'citation-unavailable'; missing.textContent = '[source unavailable]';
        missing.title = 'No retrieved passage was returned for this citation.';
        parent.append(missing);
      }
    } else {
      const tag = token.startsWith('`') ? 'code' : token.startsWith('**') || token.startsWith('__') ? 'strong' : 'em';
      const element = document.createElement(tag);
      const width = tag === 'strong' ? 2 : 1;
      element.textContent = token.slice(width, -width);
      parent.append(element);
    }
    offset = match.index + token.length;
  }
  parent.append(document.createTextNode(text.slice(offset)));
}

function markdown(parent, text, targets) {
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  let block = null;
  let list = null;
  let code = null;
  for (const line of lines) {
    if (/^\s*```/.test(line)) {
      if (code) code = null;
      else {
        const pre = document.createElement('pre'); code = document.createElement('code');
        pre.append(code); parent.append(pre);
      }
      block = null; list = null; continue;
    }
    if (code) { code.append(document.createTextNode(line + '\n')); continue; }
    if (!line.trim()) { block = null; list = null; continue; }
    const heading = line.match(/^#{1,6}\s+(.+)$/);
    const bullet = line.match(/^\s*(?:[-*+]\s+|\d+[.)]\s+)(.+)$/);
    if (heading) {
      const h = document.createElement('h3'); inline(h, heading[1], targets); parent.append(h);
      block = null; list = null;
    } else if (bullet) {
      const type = /^\s*\d/.test(line) ? 'ol' : 'ul';
      if (!list || list.tagName.toLowerCase() !== type) {
        list = document.createElement(type); parent.append(list);
      }
      const item = document.createElement('li'); inline(item, bullet[1], targets); list.append(item); block = null;
    } else {
      list = null;
      if (!block) { block = document.createElement('p'); parent.append(block); }
      else block.append(document.createTextNode(' '));
      inline(block, line, targets);
    }
  }
}

let answerSequence = 0;
function readableSection(section) {
  return section.replace(/\*\*|__/g, '').replace(/^\s*\d+(?:\.\d+)*[.)]?\s+/, '').trim() || 'Paper passage';
}

function renderAnswer(article, text, sources) {
  const targets = new Map();
  const cited = [];
  const other = [];
  const sequence = ++answerSequence;
  for (const source of sources) {
    const title = readableSection(source.section);
    const details = document.createElement('details'); details.className = 'source-card';
    details.id = `answer-${sequence}-${source.label.replace(':','-')}`;
    const summary = document.createElement('summary');
    summary.textContent = `${source.citation_number ? `[${source.citation_number}] ` : ''}${title}`;
    const documentName = document.createElement('span'); documentName.className = 'source-document';
    documentName.textContent = 'Ragas paper · Retrieved passage';
    const passage = document.createElement('div'); passage.className = 'passage';
    markdown(passage, source.text, new Map());
    details.append(summary, documentName, passage);
    if (source.cited && source.citation_number) {
      targets.set(source.label, {details, summary, title, number:source.citation_number}); cited.push(details);
    } else other.push(details);
  }
  const content = document.createElement('div'); content.className = 'answer-content';
  markdown(content, text, targets); article.append(content);
  if (cited.length) {
    const section = document.createElement('section'); section.className = 'answer-sources';
    section.setAttribute('aria-label', 'Cited sources');
    const heading = document.createElement('h3'); heading.textContent = `Sources · ${cited.length} cited`;
    section.append(heading, ...cited); article.append(section);
  }
  if (other.length) {
    const more = document.createElement('details'); more.className = 'other-sources';
    const summary = document.createElement('summary'); summary.textContent = `Other passages searched (${other.length})`;
    more.append(summary, ...other); article.append(more);
  }
}
