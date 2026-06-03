// Declarative parameter controls. spec: [{ key, label, type, min,max,step, value, options }]
// type: 'range' (slider + number, synced) | 'number' | 'toggle' | 'checks' (multi-check)
export function buildControls(container, spec) {
  container.replaceChildren();
  const state = {};
  for (const f of spec) {
    state[f.key] = f.value;
    const wrap = document.createElement('div');
    wrap.className = 'ctrl';
    const label = document.createElement('label');
    label.textContent = f.label;
    wrap.appendChild(label);

    if (f.type === 'range') {
      const row = document.createElement('div'); row.className = 'ctrl-row';
      const slider = document.createElement('input');
      slider.type = 'range'; slider.min = f.min; slider.max = f.max; slider.step = f.step; slider.value = f.value;
      const num = document.createElement('input');
      num.type = 'number'; num.min = f.min; num.max = f.max; num.step = f.step; num.value = f.value;
      const sync = (v) => { state[f.key] = Number(v); slider.value = v; num.value = v; };
      slider.oninput = () => sync(slider.value);
      num.oninput = () => sync(num.value);
      row.append(slider, num); wrap.appendChild(row);
    } else if (f.type === 'number') {
      const num = document.createElement('input');
      num.type = 'number'; if (f.min != null) num.min = f.min; num.value = f.value;
      num.oninput = () => { state[f.key] = Number(num.value); };
      wrap.appendChild(num);
    } else if (f.type === 'toggle') {
      const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = !!f.value;
      cb.onchange = () => { state[f.key] = cb.checked; };
      label.prepend(cb);
    } else if (f.type === 'select') {
      const sel = document.createElement('select');
      for (const opt of f.options) {
        const o = document.createElement('option');
        o.value = opt.value; o.textContent = opt.label;
        if (opt.disabled) { o.disabled = true; o.textContent += ' — unavailable'; }
        sel.appendChild(o);
      }
      sel.value = f.value;
      sel.onchange = () => { state[f.key] = sel.value; };
      wrap.appendChild(sel);
    } else if (f.type === 'checks') {
      state[f.key] = [...f.value];
      const row = document.createElement('div'); row.className = 'ctrl-row';
      for (const opt of f.options) {
        const cb = document.createElement('label'); cb.className = 'chip';
        const input = document.createElement('input'); input.type = 'checkbox';
        input.checked = f.value.includes(opt);
        input.onchange = () => {
          const set = new Set(state[f.key]);
          input.checked ? set.add(opt) : set.delete(opt);
          state[f.key] = [...set].sort((a, b) => a - b);
        };
        cb.append(input, document.createTextNode(' ' + opt + '°'));
        row.appendChild(cb);
      }
      wrap.appendChild(row);
    }
    container.appendChild(wrap);
  }
  return state; // live-updating value object
}
