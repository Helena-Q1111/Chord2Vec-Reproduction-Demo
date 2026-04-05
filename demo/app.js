(function () {
  const data = window.CHORD2VEC_DEMO_DATA;
  const ToneLib = window.Tone;
  const svg = document.getElementById('plot');
  const similarityCornerEl = document.getElementById('similarity-corner');

  if (!data || !Array.isArray(data.points) || !ToneLib) {
    return;
  }

  const CATEGORY_COLORS = {
    major: getComputedStyle(document.documentElement).getPropertyValue('--major').trim(),
    minor: getComputedStyle(document.documentElement).getPropertyValue('--minor').trim(),
    dominant: getComputedStyle(document.documentElement).getPropertyValue('--dominant').trim(),
    augdim: getComputedStyle(document.documentElement).getPropertyValue('--augdim').trim(),
    other: getComputedStyle(document.documentElement).getPropertyValue('--other').trim(),
  };
  const ALWAYS_HIGHLIGHT_CHORDS = new Set(['C', 'Dm', 'Em', 'F', 'G', 'Am']);

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const WIDTH = 1400;
  const HEIGHT = 980;
  const MARGIN = 64;
  const SPREAD = 1.45;
  const LEFT_CUTOFF = -0.91;
  const BASE_LABEL_FONT_SIZE = 8.5;
  const BASE_HIGHLIGHT_LABEL_FONT_SIZE = 12;

  const points = data.points
    .filter((point) => point.x >= LEFT_CUTOFF)
    .map((point) => ({ ...point }));
  inflateScatter(points, SPREAD);
  const bounds = getBounds(points);
  const scaleX = createScale(bounds.minX, bounds.maxX, MARGIN, WIDTH - MARGIN);
  const scaleY = createScale(bounds.minY, bounds.maxY, HEIGHT - MARGIN, MARGIN);

  svg.setAttribute('viewBox', `0 0 ${WIDTH} ${HEIGHT}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

  const background = createSvg('rect', {
    x: 0,
    y: 0,
    width: WIDTH,
    height: HEIGHT,
    fill: 'transparent',
  });
  svg.appendChild(background);

  const linksLayer = createSvg('g', { 'aria-hidden': 'true' });
  const pointsLayer = createSvg('g', { 'aria-hidden': 'true' });
  const labelsLayer = createSvg('g', { 'aria-hidden': 'true' });
  svg.appendChild(linksLayer);
  svg.appendChild(pointsLayer);
  svg.appendChild(labelsLayer);

  const viewport = {
    x: 0,
    y: 0,
    width: WIDTH,
    height: HEIGHT,
  };
  let zoomState = { ...viewport };
  const pointers = new Map();
  let pinchStart = null;

  const sampler = createSampler(ToneLib);
  let audioReady = false;
  let selectedPoint = null;
  let comparison = null;
  let hoveredChord = null;
  const pointNodes = new Map();

  points.forEach((point) => {
    point.x = scaleX(point.x);
    point.y = scaleY(point.y);
    point.radius = point.category === 'other' ? 5.2 : 6.0;
    point.pitch = chordToVoicing(point.chord, point.intervalSteps);

    const group = createSvg('g', { class: 'point-group' });
    const circle = createSvg('circle', {
      class: 'point-shape',
      cx: point.x,
      cy: point.y,
      r: point.radius,
      fill: CATEGORY_COLORS[point.category] || CATEGORY_COLORS.other,
      stroke: 'rgba(255,255,255,0.92)',
      'stroke-width': 1.2,
    });
    const label = createSvg('text', {
      class: 'node-label',
      x: point.x + 4,
      y: point.y - 4,
    });
    label.textContent = point.chord;
    if (isAlwaysHighlightedChord(point.chord)) {
      label.classList.add('is-always-highlighted');
    }

    group.appendChild(circle);
    pointsLayer.appendChild(group);
    labelsLayer.appendChild(label);
    pointNodes.set(point.chord, { group, circle, label, point });

    group.addEventListener('mouseenter', () => {
      hoveredChord = point.chord;
      group.classList.add('is-hover');
      label.classList.add('is-hover');
    });
    group.addEventListener('mouseleave', () => {
      if (hoveredChord === point.chord) {
        hoveredChord = null;
      }
      group.classList.remove('is-hover');
      if (
        selectedPoint?.chord !== point.chord &&
        comparison?.from?.chord !== point.chord &&
        comparison?.to?.chord !== point.chord &&
        !isAlwaysHighlightedChord(point.chord)
      ) {
        label.classList.remove('is-hover');
      }
    });
    group.addEventListener('click', () => handlePointClick(point));
  });

  svg.addEventListener('wheel', handleWheel, { passive: false });
  svg.addEventListener('pointerdown', handlePointerDown);
  svg.addEventListener('pointermove', handlePointerMove);
  svg.addEventListener('pointerup', handlePointerEnd);
  svg.addEventListener('pointercancel', handlePointerEnd);
  svg.addEventListener('pointerleave', handlePointerEnd);

  applyViewport();

  async function handlePointClick(point) {
    await unlockAudio();
    playChord(point);

    if (!selectedPoint || comparison) {
      selectedPoint = point;
      comparison = null;
      updateSelectionState();
      updateSimilarityHud();
      clearComparisonGraphics();
      return;
    }

    if (selectedPoint.chord === point.chord) {
      updateSelectionState();
      updateSimilarityHud();
      return;
    }

    comparison = {
      from: selectedPoint,
      to: point,
      cosine: cosineSimilarity(selectedPoint.embedding, point.embedding),
    };
    updateSelectionState();
    updateSimilarityHud();
    renderComparisonGraphics();
  }

  async function unlockAudio() {
    if (!audioReady) {
      await ToneLib.start();
      audioReady = true;
    }
    if (sampler.loaded) {
      await sampler.loaded;
    }
  }

  function playChord(point) {
    const notes = point.pitch;
    if (!notes.length) {
      return;
    }
    sampler.triggerAttackRelease(notes, 1.7);
  }

  function updateSelectionState() {
    pointNodes.forEach(({ group, label, point }) => {
      const pointIsSelected = selectedPoint?.chord === point.chord || comparison?.to.chord === point.chord;
      const labelIsSelected =
        selectedPoint?.chord === point.chord ||
        comparison?.from?.chord === point.chord ||
        comparison?.to?.chord === point.chord;
      const keepHover = hoveredChord === point.chord || labelIsSelected || isAlwaysHighlightedChord(point.chord);
      group.classList.toggle('is-selected', pointIsSelected);
      label.classList.toggle('is-selected', labelIsSelected);
      label.classList.toggle('is-hover', keepHover);
      group.classList.toggle('is-hover', hoveredChord === point.chord);
    });
  }

  function renderComparisonGraphics() {
    clearComparisonGraphics();
    if (!comparison) {
      return;
    }

    const line = createSvg('line', {
      class: 'link-line',
      x1: comparison.from.x,
      y1: comparison.from.y,
      x2: comparison.to.x,
      y2: comparison.to.y,
    });
    linksLayer.appendChild(line);
  }

  function clearComparisonGraphics() {
    linksLayer.replaceChildren();
  }

  function updateSimilarityHud() {
    if (!similarityCornerEl) {
      return;
    }
    if (!comparison || !Number.isFinite(comparison.cosine)) {
      similarityCornerEl.textContent = 'Cosine: -';
      return;
    }
    similarityCornerEl.textContent = `Cosine: ${comparison.cosine.toFixed(3)}`;
  }

  function handleWheel(event) {
    event.preventDefault();
    const factor = event.deltaY < 0 ? 0.88 : 1.12;
    zoomAt(clientToSvgPoint(event.clientX, event.clientY), factor);
  }

  function handlePointerDown(event) {
    if (event.pointerType !== 'touch') {
      return;
    }
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pointers.size === 2) {
      pinchStart = buildPinchSnapshot();
    }
  }

  function handlePointerMove(event) {
    if (event.pointerType !== 'touch') {
      return;
    }
    if (!pointers.has(event.pointerId)) {
      return;
    }
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pointers.size === 2 && pinchStart) {
      const snapshot = buildPinchSnapshot();
      if (!snapshot) {
        return;
      }
      const factor = pinchStart.distance / (snapshot.distance || pinchStart.distance || 1);
      applyPinchZoom(pinchStart, snapshot.center, factor);
    }
  }

  function handlePointerEnd(event) {
    if (event.pointerType !== 'touch') {
      return;
    }
    if (pointers.has(event.pointerId)) {
      pointers.delete(event.pointerId);
    }
    if (pointers.size < 2) {
      pinchStart = null;
    }
  }

  function buildPinchSnapshot() {
    const entries = [...pointers.values()];
    if (entries.length !== 2) {
      return null;
    }
    const a = clientToSvgPoint(entries[0].x, entries[0].y);
    const b = clientToSvgPoint(entries[1].x, entries[1].y);
    return {
      center: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 },
      distance: Math.hypot(a.x - b.x, a.y - b.y),
    };
  }

  function applyPinchZoom(startSnapshot, currentCenter, factor) {
    const nextWidth = clamp(zoomState.width * factor, viewport.width * 0.42, viewport.width * 4.5);
    const nextHeight = clamp(zoomState.height * factor, viewport.height * 0.42, viewport.height * 4.5);
    zoomState = {
      width: nextWidth,
      height: nextHeight,
      x: currentCenter.x - nextWidth / 2,
      y: currentCenter.y - nextHeight / 2,
    };
    applyViewport();
  }

  function zoomAt(center, factor) {
    const nextWidth = clamp(zoomState.width * factor, viewport.width * 0.42, viewport.width * 4.5);
    const nextHeight = clamp(zoomState.height * factor, viewport.height * 0.42, viewport.height * 4.5);
    zoomState = {
      width: nextWidth,
      height: nextHeight,
      x: center.x - (center.x - zoomState.x) * (nextWidth / zoomState.width),
      y: center.y - (center.y - zoomState.y) * (nextHeight / zoomState.height),
    };
    applyViewport();
  }

  function applyViewport() {
    svg.setAttribute('viewBox', `${zoomState.x} ${zoomState.y} ${zoomState.width} ${zoomState.height}`);
    updateLabelVisualScale();
  }

  function updateLabelVisualScale() {
    const ratio = zoomState.width / viewport.width;
    pointNodes.forEach(({ label, point }) => {
      const baseSize = isAlwaysHighlightedChord(point.chord)
        ? BASE_HIGHLIGHT_LABEL_FONT_SIZE
        : BASE_LABEL_FONT_SIZE;
      label.style.fontSize = `${(baseSize * ratio).toFixed(3)}px`;
    });
  }

  function clientToSvgPoint(clientX, clientY) {
    const rect = svg.getBoundingClientRect();
    return {
      x: zoomState.x + ((clientX - rect.left) / rect.width) * zoomState.width,
      y: zoomState.y + ((clientY - rect.top) / rect.height) * zoomState.height,
    };
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function isAlwaysHighlightedChord(chord) {
    if (!chord) {
      return false;
    }
    const normalized = normalizeChordName(chord);
    return ALWAYS_HIGHLIGHT_CHORDS.has(normalized);
  }

  function normalizeChordName(chord) {
    const name = String(chord).trim();
    const lower = name.toLowerCase();
    if (lower === 'cmaj') {
      return 'C';
    }
    if (lower === 'dmin') {
      return 'Dm';
    }
    if (lower === 'em') {
      return 'Em';
    }
    if (lower === 'f') {
      return 'F';
    }
    if (lower === 'g') {
      return 'G';
    }
    if (lower === 'am') {
      return 'Am';
    }
    return name;
  }

  function cosineSimilarity(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
      return NaN;
    }
    let dot = 0;
    let magA = 0;
    let magB = 0;
    for (let i = 0; i < a.length; i += 1) {
      dot += a[i] * b[i];
      magA += a[i] * a[i];
      magB += b[i] * b[i];
    }
    const denom = Math.sqrt(magA) * Math.sqrt(magB);
    return denom > 0 ? dot / denom : NaN;
  }

  function createSampler(Tone) {
    return new Tone.Sampler({
      urls: {
        A0: 'A0.mp3',
        C1: 'C1.mp3',
        'D#1': 'Ds1.mp3',
        'F#1': 'Fs1.mp3',
        A1: 'A1.mp3',
        C2: 'C2.mp3',
        'D#2': 'Ds2.mp3',
        'F#2': 'Fs2.mp3',
        A2: 'A2.mp3',
        C3: 'C3.mp3',
        'D#3': 'Ds3.mp3',
        'F#3': 'Fs3.mp3',
        A3: 'A3.mp3',
        C4: 'C4.mp3',
        'D#4': 'Ds4.mp3',
        'F#4': 'Fs4.mp3',
        A4: 'A4.mp3',
        C5: 'C5.mp3',
        'D#5': 'Ds5.mp3',
        'F#5': 'Fs5.mp3',
        A5: 'A5.mp3',
        C6: 'C6.mp3',
        'D#6': 'Ds6.mp3',
        'F#6': 'Fs6.mp3',
        A6: 'A6.mp3',
      },
      release: 1.4,
      baseUrl: 'https://tonejs.github.io/audio/salamander/',
    }).toDestination();
  }

  function chordToVoicing(symbol, intervalSteps) {
    const match = symbol.match(/^([A-G](?:#|b)?)(.*)$/);
    if (!match) {
      return [];
    }

    const root = match[1];
    const quality = match[2] || '';
    const octave = 3;
    const rootNote = `${root}${octave}`;
    const derivedOffsets = intervalStepsToOffsets(intervalSteps);
    const intervals = derivedOffsets.length > 0 ? derivedOffsets : getSemitoneOffsets(quality);
    return intervals.map((interval) => ToneLib.Frequency(rootNote).transpose(interval).toNote());
  }

  function intervalStepsToOffsets(intervalSteps) {
    if (!Array.isArray(intervalSteps) || intervalSteps.length === 0) {
      return [];
    }
    const offsets = [0];
    let running = 0;
    intervalSteps.forEach((step) => {
      const numeric = Number(step);
      if (!Number.isFinite(numeric) || numeric <= 0) {
        return;
      }
      running += numeric;
      offsets.push(running);
    });
    return normalizeOffsets(offsets);
  }

  function getSemitoneOffsets(quality) {
    const q = String(quality || '');

    const unknownOffsets = parseUnknownIntervalOffsets(q);
    if (unknownOffsets.length > 0) {
      return normalizeOffsets(unknownOffsets);
    }

    if (q === 'pedal') {
      return [0];
    }

    const offsets = [0];
    const lower = q.toLowerCase();

    if (lower.includes('sus4')) {
      offsets.push(5, 7);
    } else if (lower.includes('sus2')) {
      offsets.push(2, 7);
    } else if (lower.includes('dim')) {
      offsets.push(3, 6);
    } else if (lower.includes('aug')) {
      offsets.push(4, 8);
    } else if (lower.startsWith('m') && !lower.startsWith('maj')) {
      offsets.push(3, 7);
    } else if (lower.includes('5') && !lower.includes('maj') && !lower.includes('m7b5')) {
      offsets.push(7);
    } else {
      offsets.push(4, 7);
    }

    if (lower.includes('no3')) {
      removeOffset(offsets, 3);
      removeOffset(offsets, 4);
      removeOffset(offsets, 5);
    }

    if (lower.includes('m7b5')) {
      removeOffset(offsets, 7);
      offsets.push(6, 10);
    }
    if (lower.includes('dim7')) {
      removeOffset(offsets, 10);
      offsets.push(9);
    } else if (lower.includes('maj7') || lower.includes('mmaj7') || lower.includes('augmaj7')) {
      offsets.push(11);
    } else if (/(^|[^a-z])7/.test(lower) || lower.endsWith('7') || lower.includes('sus7')) {
      offsets.push(10);
    }

    if (lower.includes('6') && !lower.includes('16') && !lower.includes('13')) {
      offsets.push(9);
    }

    if (lower.includes('b9')) {
      offsets.push(13);
    }
    if (lower.includes('#9')) {
      offsets.push(15);
    }
    if (lower.includes('9')) {
      offsets.push(14);
    }
    if (lower.includes('11')) {
      offsets.push(17);
    }
    if (lower.includes('#11')) {
      offsets.push(18);
    }
    if (lower.includes('13')) {
      offsets.push(21);
    }
    if (lower.includes('b13')) {
      offsets.push(20);
    }
    if (lower.includes('add9')) {
      offsets.push(14);
    }
    if (lower.includes('add11')) {
      offsets.push(17);
    }
    if (lower.includes('add13')) {
      offsets.push(21);
    }

    return normalizeOffsets(offsets);
  }

  function parseUnknownIntervalOffsets(quality) {
    const match = quality.match(/^unk\[([^\]]+)\]$/i);
    if (!match) {
      return [];
    }
    const raw = match[1].trim();
    if (!raw || raw === 'empty') {
      return [0];
    }
    const steps = raw
      .split(',')
      .map((part) => Number.parseInt(part.trim(), 10))
      .filter((value) => Number.isFinite(value) && value > 0);
    if (steps.length === 0) {
      return [0];
    }

    // Hooktheory stores root-position step intervals: [4,3,4] -> 0,4,7,11
    const offsets = [0];
    let running = 0;
    steps.forEach((step) => {
      running += step;
      offsets.push(running);
    });
    return offsets;
  }

  function normalizeOffsets(offsets) {
    const unique = [...new Set(offsets.filter((v) => Number.isFinite(v) && v >= 0))].sort((a, b) => a - b);
    return unique.length > 0 ? unique : [0, 4, 7];
  }

  function removeOffset(offsets, value) {
    const index = offsets.indexOf(value);
    if (index >= 0) {
      offsets.splice(index, 1);
    }
  }

  function createScale(min, max, outMin, outMax) {
    const span = max - min || 1;
    return (value) => outMin + ((value - min) / span) * (outMax - outMin);
  }

  function getBounds(items) {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    items.forEach((item) => {
      minX = Math.min(minX, item.x);
      maxX = Math.max(maxX, item.x);
      minY = Math.min(minY, item.y);
      maxY = Math.max(maxY, item.y);
    });
    if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
      return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
    }
    const padX = (maxX - minX) * 0.08 || 1;
    const padY = (maxY - minY) * 0.08 || 1;
    return {
      minX: minX - padX,
      maxX: maxX + padX,
      minY: minY - padY,
      maxY: maxY + padY,
    };
  }

  function inflateScatter(items, spread) {
    let centerX = 0;
    let centerY = 0;
    items.forEach((item) => {
      centerX += item.x;
      centerY += item.y;
    });
    centerX /= items.length || 1;
    centerY /= items.length || 1;

    items.forEach((item) => {
      item.x = centerX + (item.x - centerX) * spread;
      item.y = centerY + (item.y - centerY) * spread;
    });
  }

  function createSvg(tag, attrs = {}) {
    const element = document.createElementNS(SVG_NS, tag);
    Object.entries(attrs).forEach(([key, value]) => {
      element.setAttribute(key, String(value));
    });
    return element;
  }
})();