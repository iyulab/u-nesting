// Custom parts editor. Edits the current geometries array (quantity / delete)
// and supports loading from JSON. Calls setGeometries on change.
export function buildPartsEditor(container, getGeometries, setGeometries) {
  function render() {
    container.replaceChildren();
    const geoms = getGeometries();
    if (!geoms.length) {
      const empty = document.createElement('div');
      empty.className = 'parts-empty';
      empty.textContent = 'No parts';
      container.appendChild(empty);
      return;
    }
    geoms.forEach((g, idx) => {
      const row = document.createElement('div'); row.className = 'part-row';
      const name = document.createElement('span'); name.textContent = g.id; name.className = 'part-id';
      const qty = document.createElement('input');
      qty.type = 'number'; qty.min = 1; qty.value = g.quantity ?? 1; qty.className = 'part-qty';
      qty.oninput = () => { geoms[idx].quantity = Math.max(1, Number(qty.value)); setGeometries(geoms); };
      const del = document.createElement('button'); del.textContent = '×'; del.className = 'part-del'; del.title = 'Remove';
      del.onclick = () => { geoms.splice(idx, 1); setGeometries(geoms); render(); };
      row.append(name, qty, del);
      container.appendChild(row);
    });
  }
  render();
  return {
    refresh: render,
    loadJson(text) {
      const data = JSON.parse(text);
      if (!Array.isArray(data.geometries)) throw new Error('A "geometries" array is required');
      setGeometries(data.geometries);
      render();
      return data.boundary || null;
    },
  };
}
