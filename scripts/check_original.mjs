// Compare Python slides and score increments with the original JS implementation.
import fs from 'node:fs';
import vm from 'node:vm';
import readline from 'node:readline';
const context = vm.createContext({ console });
for (const name of ['tile.js', 'grid.js', 'game_manager.js']) {
  vm.runInContext(fs.readFileSync(new URL(`../vendor/original-2048/${name}`, import.meta.url), 'utf8'), context);
}
for await (const line of readline.createInterface({ input: process.stdin })) {
  const { board, action } = JSON.parse(line);
  const manager = Object.create(context.GameManager.prototype);
  manager.size = 4;
  manager.grid = new context.Grid(4);
  manager.score = 0;
  manager.over = false;
  manager.won = false;
  manager.keepPlaying = true;
  manager.actuate = () => {};
  manager.addRandomTile = () => {};
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
    if (board[4 * r + c]) manager.grid.insertTile(new context.Tile({ x: c, y: r }, board[4 * r + c]));
  }
  manager.move({ UP: 0, RIGHT: 1, DOWN: 2, LEFT: 3 }[action]);
  const result = [];
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
    result.push(manager.grid.cells[c][r]?.value ?? 0);
  }
  process.stdout.write(JSON.stringify({ board: result, score: manager.score }) + '\n');
}
