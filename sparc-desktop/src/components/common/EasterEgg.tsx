import { useState, useEffect, useRef, useCallback } from "react";

interface Point { x: number; y: number }

const CELL = 16;
const COLS = 20;
const ROWS = 15;
const W = COLS * CELL;
const H = ROWS * CELL;
const TICK_MS = 120;

const DIRECTIONS: Record<string, Point> = {
  ArrowUp: { x: 0, y: -1 },
  ArrowDown: { x: 0, y: 1 },
  ArrowLeft: { x: -1, y: 0 },
  ArrowRight: { x: 1, y: 0 },
};

function randomFood(snake: Point[]): Point {
  let p: Point;
  do {
    p = { x: Math.floor(Math.random() * COLS), y: Math.floor(Math.random() * ROWS) };
  } while (snake.some((s) => s.x === p.x && s.y === p.y));
  return p;
}

interface Props {
  onClose: () => void;
}

export default function EasterEgg({ onClose }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dirRef = useRef<Point>({ x: 1, y: 0 });
  const nextDirRef = useRef<Point>({ x: 1, y: 0 });
  const [score, setScore] = useState(0);
  const [gameOver, setGameOver] = useState(false);
  const snakeRef = useRef<Point[]>([{ x: 3, y: 7 }, { x: 2, y: 7 }, { x: 1, y: 7 }]);
  const foodRef = useRef<Point>(randomFood(snakeRef.current));
  const tickRef = useRef<number | null>(null);

  const draw = useCallback(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    // background
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, W, H);
    // grid
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    for (let x = 0; x <= W; x += CELL) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y <= H; y += CELL) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
    // food
    ctx.fillStyle = "#e94d9b";
    ctx.fillRect(foodRef.current.x * CELL + 2, foodRef.current.y * CELL + 2, CELL - 4, CELL - 4);
    // snake
    snakeRef.current.forEach((s, i) => {
      ctx.fillStyle = i === 0 ? "#602468" : "#9e337d";
      ctx.fillRect(s.x * CELL + 1, s.y * CELL + 1, CELL - 2, CELL - 2);
    });
  }, []);

  const tick = useCallback(() => {
    dirRef.current = nextDirRef.current;
    const head = snakeRef.current[0];
    const nx = head.x + dirRef.current.x;
    const ny = head.y + dirRef.current.y;

    // collision check
    if (nx < 0 || nx >= COLS || ny < 0 || ny >= ROWS || snakeRef.current.some((s) => s.x === nx && s.y === ny)) {
      setGameOver(true);
      return;
    }

    const newHead = { x: nx, y: ny };
    const newSnake = [newHead, ...snakeRef.current];

    if (nx === foodRef.current.x && ny === foodRef.current.y) {
      foodRef.current = randomFood(newSnake);
      setScore((s) => s + 10);
    } else {
      newSnake.pop();
    }

    snakeRef.current = newSnake;
    draw();
  }, [draw]);

  const restart = () => {
    snakeRef.current = [{ x: 3, y: 7 }, { x: 2, y: 7 }, { x: 1, y: 7 }];
    dirRef.current = { x: 1, y: 0 };
    nextDirRef.current = { x: 1, y: 0 };
    foodRef.current = randomFood(snakeRef.current);
    setScore(0);
    setGameOver(false);
  };

  useEffect(() => {
    if (gameOver) return;
    tickRef.current = window.setInterval(tick, TICK_MS);
    return () => { if (tickRef.current) clearInterval(tickRef.current); };
  }, [gameOver, tick]);

  useEffect(() => {
    draw();
    const onKey = (e: KeyboardEvent) => {
      const d = DIRECTIONS[e.key];
      if (!d) return;
      e.preventDefault();
      // prevent reversing
      if (d.x !== -dirRef.current.x || d.y !== -dirRef.current.y) {
        nextDirRef.current = d;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [draw]);

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70">
      <div className="retro-box bg-[#1a1a2e] p-4 text-center" style={{ margin: "12px" }}>
        <div className="flex items-center justify-between mb-2">
          <span
            className="text-xs font-bold tracking-widest text-sparc-pink"
            style={{ fontFamily: '"Courier New", monospace' }}
          >
            ▸ SPARC SNAKE
          </span>
          <span className="text-xs text-sparc-gold" style={{ fontFamily: '"Courier New", monospace' }}>
            SCORE: {score}
          </span>
        </div>
        <canvas ref={canvasRef} width={W} height={H} className="border border-sparc-purple/40" />
        {gameOver && (
          <div className="mt-3 space-y-2">
            <p className="text-xs text-sparc-crimson" style={{ fontFamily: '"Courier New", monospace' }}>
              GAME OVER — SCORE: {score}
            </p>
            <div className="flex justify-center gap-2">
              <button
                onClick={restart}
                className="retro-toggle rounded bg-sparc-purple px-3 py-1 text-xs text-white hover:bg-sparc-magenta"
              >
                RETRY
              </button>
              <button
                onClick={onClose}
                className="retro-toggle rounded border border-sparc-gray-600 px-3 py-1 text-xs text-sparc-gray-300 hover:bg-sparc-gray-800"
              >
                CLOSE
              </button>
            </div>
          </div>
        )}
        {!gameOver && (
          <p className="mt-2 text-xs text-sparc-gray-600" style={{ fontFamily: '"Courier New", monospace' }}>
            Arrow keys to move · ESC to close
          </p>
        )}
      </div>
    </div>
  );
}
