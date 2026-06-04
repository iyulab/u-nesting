import { test } from 'node:test';
import assert from 'node:assert/strict';
import { colorFor, seedColors } from '../render2d.js';

// Regression: a part's color must depend only on its position in the current geometry
// set, not on which sets were rendered earlier in the session. A prior global cache
// leaked across mode switches (e.g. 3D's ids shifted 2D's A/B/C down the palette), so
// the same part changed color between modes. seedColors must reset that mapping.
test('seedColors makes colors order-independent across renders', () => {
  seedColors([{ id: 'A' }, { id: 'B' }, { id: 'C' }]);
  const a1 = colorFor('A');
  const b1 = colorFor('B');

  // A different geometry set is rendered (another mode/preset).
  seedColors([{ id: 'big' }, { id: 'mid' }, { id: 'small' }]);
  assert.equal(colorFor('big'), a1); // first slot is always the first palette color

  // Returning to the original set restores the original colors.
  seedColors([{ id: 'A' }, { id: 'B' }, { id: 'C' }]);
  assert.equal(colorFor('A'), a1);
  assert.equal(colorFor('B'), b1);
});

test('distinct ids in one set get distinct colors', () => {
  seedColors([{ id: 'A' }, { id: 'B' }, { id: 'C' }]);
  const colors = new Set(['A', 'B', 'C'].map(colorFor));
  assert.equal(colors.size, 3);
});
