// script.js
// Connect 4 with Pure MCTS AI (Monte-Carlo Tree Search)
// Uses UCB1 for exploration-exploitation balance

// =================== Config ===================
const ROWS = 6;
const COLS = 7;

const BASE_SIMULATIONS = 250;   // base number of MCTS playouts (scaled by rating)
const SIMULATION_LIMIT = 1200;  // hard cap on total simulations
const MCTS_PLY_DEPTH = 12;      // limit rollout length (avoid super long playouts)
const RANDOM_MOVE_TIEBREAK = true;
const UCB_EXPLORATION = Math.sqrt(2); // UCB1 exploration constant
const FORCED_WIN_SCORE = 1000000;
const MAX_GAME_PLIES = ROWS * COLS;
const SEARCH_EVAL_CAP = 9000;
const SEARCH_INF = FORCED_WIN_SCORE + 1000;
const SEARCH_MOVE_ORDER = [3,2,4,1,5,0,6];
const SEARCH_TT_EXACT = 0;
const SEARCH_TT_LOWER = 1;
const SEARCH_TT_UPPER = 2;
const ANALYSIS_DB_NAME = 'cn4-analysis-cache';
const ANALYSIS_DB_VERSION = 1;
const ANALYSIS_STORE_NAME = 'positions';

// =================== State ===================
let board = [];
let currentPlayer = 1; // 1 = A (cyan), 2 = B (red)
let gameOver = false;
let lastWinningCells = null;
let moveHistory = [];
let pendingAITimeout = null; // track pending AI move for cancellation
let analysisWorker = null;
let analysisJobId = 0;
let latestAnalysis = null;
let analysisDbPromise = null;
const analysisMemory = new Map();

// =================== DOM ===================
const boardEl = document.getElementById('board');
const overlayEl = document.getElementById('col-overlay');
const messageEl = document.getElementById('message');
const restartBtn = document.getElementById('restart');

const aiToggle = document.getElementById('ai-toggle');
const aiPlayerSelect = document.getElementById('ai-player');
const aiRatingInput = document.getElementById('ai-rating');
const aiClassLabel = document.getElementById('ai-class');

const evalBar = document.getElementById('eval-bar');
const evalLeft = document.getElementById('eval-left');
const evalRight = document.getElementById('eval-right');
const evalLabel = document.getElementById('eval-label');
const evalMeta = document.getElementById('eval-meta');

const threatsEl = document.getElementById('threats');
const showReviewBtn = document.getElementById('show-review');
const reviewEl = document.getElementById('review');

// =================== Init helpers ===================
function createEmptyBoard(){ board = Array.from({length: ROWS}, ()=>Array(COLS).fill(0)); }
function cloneBoard(b){ return b.map(r => r.slice()); }

function inBounds(r,c){ return r>=0 && r<ROWS && c>=0 && c<COLS; }

function getValidMoves(bd){
  const moves = [];
  for(let c=0;c<COLS;c++) if(bd[0][c]===0) moves.push(c);
  return moves;
}
function applyMove(bd, col, player){
  const nb = cloneBoard(bd);
  for(let r=ROWS-1;r>=0;r--) if(nb[r][col]===0){ nb[r][col]=player; return {board: nb, row: r}; }
  return null;
}

// =================== Rendering ===================
function renderBoard(){
  boardEl.innerHTML = '';
  overlayEl.innerHTML = '';
  for(let r=0;r<ROWS;r++){
    for(let c=0;c<COLS;c++){
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.row = r; cell.dataset.col = c;
      const disc = document.createElement('div');
      disc.className = 'disc';
      const val = board[r][c];
      if(val===0) disc.classList.add('empty');
      else disc.classList.add(val===1 ? 'playera' : 'playerb');
      cell.appendChild(disc);
      // clicking cell drops a piece in that column
      cell.addEventListener('click', ()=> handleColumnClick(c));
      boardEl.appendChild(cell);
    }
  }
  // overlay buttons for wide screens
  for(let c=0;c<COLS;c++){
    const btn = document.createElement('button');
    btn.className = 'col-btn';
    btn.dataset.col = c;
    btn.title = `Drop in column ${c+1}`;
    btn.addEventListener('click', (e)=>{ e.stopPropagation(); handleColumnClick(c);});
    overlayEl.appendChild(btn);
  }
  updateMessage();
  updateEvalAndThreats();
}

// =================== Messages ===================
function updateMessage(text){
  if(text){ messageEl.textContent = text; return; }
  if(gameOver) return;
  const name = currentPlayer===1 ? 'A' : 'B';
  const color = currentPlayer===1 ? 'Cyan' : 'Red';
  messageEl.textContent = `Player ${name}'s turn (${color})`;
}

// =================== Win detection ===================
function localCheckWin(bd, row, col, player){
  const dirs = [{dr:0,dc:1},{dr:1,dc:0},{dr:1,dc:1},{dr:1,dc:-1}];
  for(const {dr,dc} of dirs){
    let count=1;
    let r=row+dr, c=col+dc;
    while(inBounds(r,c) && bd[r][c]===player){ count++; r+=dr; c+=dc; }
    r=row-dr; c=col-dc;
    while(inBounds(r,c) && bd[r][c]===player){ count++; r-=dr; c-=dc; }
    if(count>=4) return true;
  }
  return false;
}
function checkWin(row,col,player){
  const dirs = [{dr:0,dc:1},{dr:1,dc:0},{dr:1,dc:1},{dr:1,dc:-1}];
  for(const {dr,dc} of dirs){
    let count=1; const cells=[[row,col]];
    let r=row+dr, c=col+dc;
    while(inBounds(r,c) && board[r][c]===player){ cells.push([r,c]); count++; r+=dr; c+=dc; }
    r=row-dr; c=col-dc;
    while(inBounds(r,c) && board[r][c]===player){ cells.push([r,c]); count++; r-=dr; c-=dc; }
    if(count>=4){ lastWinningCells = cells; return true; }
  }
  lastWinningCells = null;
  return false;
}
function highlightWinningLine(){
  if(!lastWinningCells) return;
  for(const [r,c] of lastWinningCells){
    const idx = r*COLS + c;
    const disc = boardEl.children[idx]?.querySelector('.disc');
    if(disc){ disc.style.outline='4px solid rgba(255,255,255,0.18)'; disc.style.transform='scale(1.06)'; }
  }
}

// =================== Brute-Force Evaluation & Winning Path Detection ===================
// Detect if a player has a guaranteed winning path (cannot be blocked by opponent)
// This uses exhaustive game-tree analysis up to a limited depth
// WITH TIMEOUT protection to prevent UI freezing
function hasGuaranteedWinPath(bd, player, depth=0, maxDepth=4, startTime=null) {
  // Add timeout protection - max 50ms per evaluation
  if (startTime === null) startTime = Date.now();
  if (Date.now() - startTime > 50) return false; // timeout
  
  // check if player has already won
  if (depth > 0) {
    const valid = getValidMoves(bd);
    if (valid.length === 0) return false; // draw
  }
  
  // limit depth to prevent infinite recursion
  if (depth >= maxDepth) return false;
  
  // check immediate winning moves
  const winCols = findImmediateWinningColsFor(bd, player);
  if (winCols.length > 0) return true;
  
  // try each move and see if we can force a win
  const opponent = player === 1 ? 2 : 1;
  const validMoves = getValidMoves(bd);
  
  if (validMoves.length === 0) return false;
  
  // OPTIMIZATION: only check first few moves to save time
  const moveLimit = depth === 0 ? Math.min(3, validMoves.length) : validMoves.length;
  
  // count forcing paths: moves where player can win regardless of opponent's best response
  let forcingPaths = 0;
  
  for (let i = 0; i < moveLimit; i++) {
    const col = validMoves[i];
    const res = applyMove(bd, col, player);
    if (!res) continue;
    
    if (localCheckWin(res.board, res.row, col, player)) {
      forcingPaths++;
      continue;
    }
    
    // if opponent blocks here, can we still win?
    const oppWinCols = findImmediateWinningColsFor(res.board, opponent);
    let playerCanStillWin = true;
    
    if (oppWinCols.length > 0) {
      // opponent has winning move - if they take it, we need to still have a path
      for (const oppCol of oppWinCols) {
        const oppRes = applyMove(res.board, oppCol, opponent);
        if (!oppRes) continue;
        
        if (localCheckWin(oppRes.board, oppRes.row, oppCol, opponent)) {
          // opponent wins here
          playerCanStillWin = false;
          break;
        }
        
        // after opponent moves, can we still get guaranteed win?
        if (!hasGuaranteedWinPath(oppRes.board, player, depth + 2, maxDepth, startTime)) {
          playerCanStillWin = false;
          break;
        }
      }
    } else {
      // opponent has no immediate win, recursively check
      playerCanStillWin = hasGuaranteedWinPath(res.board, player, depth + 1, maxDepth, startTime);
    }
    
    if (playerCanStillWin) {
      forcingPaths++;
    }
  }
  
  // if we have at least one forcing path, we have a guaranteed win
  return forcingPaths > 0;
}

// Ruminate: simulate games pseudo-randomly to build evaluation statistics
// With higher iteration counts simulating massive exploration (theoretically 10^10 games per ms)
function ruminatePositionForPlayer(bd, toPlay, targetPlayer, iterations=1000) {
  let playerWins = 0;
  
  // Brute-force pseudo-simulations: each iteration represents multiple parallel universe outcomes
  // OPTIMIZATION: reduce iterations for empty/near-empty boards
  const moveCount = getValidMoves(bd).length;
  const actualIterations = moveCount > 5 ? iterations : Math.min(100, iterations);
  
  for (let i = 0; i < actualIterations; i++) {
    // run one random playout
    try {
      const result = randomPlayout(bd, toPlay);
      if (result === targetPlayer) playerWins++;
    } catch (e) {
      // safety: if playout errors, skip
    }
  }
  
  // convert to win rate percentage
  const winRate = playerWins / actualIterations;
  return Math.round(winRate * 100);
}
function ruminatePosition(bd, player, iterations=1000) {
  return ruminatePositionForPlayer(bd, player, player, iterations);
}

function computePatternScore(bd){
  const SCORE = {4:10000, 3:500, 2:30};
  let patternScore = 0;

  function scoreLine(cells){
    const counts = {0:0,1:0,2:0};
    for(const [r,c] of cells) counts[bd[r][c]]++;
    if(counts[1] && counts[2]) return 0; // blocked line
    if(counts[2]===0 && counts[1]>0) return SCORE[counts[1]] || 0;
    if(counts[1]===0 && counts[2]>0) return -(SCORE[counts[2]] || 0);
    return 0;
  }

  // horizontal
  for(let r=0;r<ROWS;r++) for(let c=0;c<=COLS-4;c++) patternScore += scoreLine([[r,c],[r,c+1],[r,c+2],[r,c+3]]);
  // vertical
  for(let c=0;c<COLS;c++) for(let r=0;r<=ROWS-4;r++) patternScore += scoreLine([[r,c],[r+1,c],[r+2,c],[r+3,c]]);
  // diag down-right
  for(let r=0;r<=ROWS-4;r++) for(let c=0;c<=COLS-4;c++) patternScore += scoreLine([[r,c],[r+1,c+1],[r+2,c+2],[r+3,c+3]]);
  // diag down-left
  for(let r=0;r<=ROWS-4;r++) for(let c=3;c<COLS;c++) patternScore += scoreLine([[r,c],[r+1,c-1],[r+2,c-2],[r+3,c-3]]);

  // center column bias
  const center = Math.floor(COLS/2);
  for(let r=0;r<ROWS;r++){
    if(bd[r][center]===1) patternScore += 12;
    if(bd[r][center]===2) patternScore -= 12;
  }

  return patternScore;
}

function computeImmediateTurnRaw(bd, toPlay){
  const opponent = toPlay === 1 ? 2 : 1;
  const toPlayWins = findImmediateWinningColsFor(bd, toPlay).length;
  if (toPlayWins > 0) return (toPlay === 1 ? 1 : -1) * (50000 + (toPlayWins * 10000));

  const opponentWins = findImmediateWinningColsFor(bd, opponent).length;
  if (opponentWins > 0) return (opponent === 1 ? 1 : -1) * (50000 + (opponentWins * 10000));

  return null;
}

function computeRuminationIterations(rating){
  const ruminationBase = Math.min(500, Math.max(100, rating / 3));
  return Math.min(1000, Math.max(100, Math.floor(ruminationBase * (rating / 1500))));
}

// Evaluate position using brute-force: guaranteed paths + rumination + pattern scoring
// player parameter: 1 for player A (cyan), 2 for player B (red) - used for perspective
// rating parameter: higher rating means deeper brute-force analysis
// useFastMode: skip expensive guaranteed win check if true to prevent UI freezing
function evaluateBoard(bd, player=1, rating=1500, useFastMode=false){
  const opponent = player === 1 ? 2 : 1;
  
  // Scale max depth based on rating
  const maxDepth = Math.min(5, Math.max(3, Math.floor(rating / 400)));
  
  // 1. Check for guaranteed winning paths (highest priority - exhaustive search)
  // SKIP in fast mode to prevent UI freezing
  let playerHasGuaranteedWin = false;
  let opponentHasGuaranteedWin = false;
  
  if (!useFastMode) {
    playerHasGuaranteedWin = hasGuaranteedWinPath(bd, player, 0, maxDepth);
    opponentHasGuaranteedWin = hasGuaranteedWinPath(bd, opponent, 0, maxDepth);
  }
  
  if (playerHasGuaranteedWin) return 100000; // guaranteed win for player
  if (opponentHasGuaranteedWin) return -100000; // opponent has guaranteed win
  
  // 2. Check immediate threat levels
  const playerWins = findImmediateWinningColsFor(bd, player).length;
  const opponentWins = findImmediateWinningColsFor(bd, opponent).length;
  
  if (playerWins > 0) return 50000 + (playerWins * 10000);
  if (opponentWins > 0) return -(50000 + (opponentWins * 10000));
  
  // 3. Ruminate: simulate games to get win probability (brute-force estimation)
  // Higher rating = more thorough rumination
  const ruminationIterations = computeRuminationIterations(rating);
  const playerWinRate = ruminatePosition(bd, player, ruminationIterations);
  
  // convert win rate (0-100) to evaluation score (-100000 to +100000)
  const ruminationScore = (playerWinRate - 50) * 4000;
  
  // 4. Pattern-based scoring (tie-breaker)
  const patternScore = computePatternScore(bd);

  // 5. Combine: rumination dominates, pattern is tie-breaker
  const score = ruminationScore + Math.min(1000, Math.max(-1000, patternScore));
  
  return score;
}

function evaluateBoardForCyan(bd, toPlay=1, rating=1500){
  const forcedRaw = computeImmediateTurnRaw(bd, toPlay);
  if (forcedRaw !== null) return forcedRaw;

  const cyanWinRate = ruminatePositionForPlayer(bd, toPlay, 1, computeRuminationIterations(rating));
  const ruminationScore = (cyanWinRate - 50) * 4000;
  const patternScore = computePatternScore(bd);

  return ruminationScore + Math.min(1000, Math.max(-1000, patternScore));
}

function findImmediateWinningColsFor(bd, player){
  const wins=[];
  for(let c=0;c<COLS;c++){
    if(bd[0][c]!==0) continue;
    for(let r=ROWS-1;r>=0;r--) if(bd[r][c]===0){
      const nb = cloneBoard(bd); nb[r][c]=player;
      if(localCheckWin(nb,r,c,player)) wins.push(c);
      break;
    }
  }
  return wins;
}

function countFilledCells(bd){
  let filled = 0;
  for(let r=0;r<ROWS;r++) for(let c=0;c<COLS;c++) if(bd[r][c]!==0) filled++;
  return filled;
}

function toCyanScore(score, rootPlayer){
  return rootPlayer === 1 ? score : -score;
}

function staticSearchEval(bd, toPlay){
  const opponent = toPlay === 1 ? 2 : 1;
  const ownWins = findImmediateWinningColsFor(bd, toPlay).length;
  const oppWins = findImmediateWinningColsFor(bd, opponent).length;
  const signedPattern = (toPlay === 1 ? 1 : -1) * computePatternScore(bd);
  return Math.max(-SEARCH_EVAL_CAP, Math.min(SEARCH_EVAL_CAP, signedPattern + (ownWins * 2200) - (oppWins * 2600)));
}

function boardKeyForSearch(bd, toPlay){
  let key = '' + toPlay;
  for(let r=0;r<ROWS;r++) for(let c=0;c<COLS;c++) key += bd[r][c];
  return key;
}

function getAnalysisPositionKey(bd, toPlay){
  return boardKeyForSearch(bd, toPlay);
}

function normalizePersistedAnalysis(entry){
  if (!entry || typeof entry !== 'object' || !entry.key) return null;
  return {
    key: String(entry.key),
    toPlay: Number(entry.toPlay) === 2 ? 2 : 1,
    raw: Number(entry.raw) || 0,
    depth: Math.max(0, Number(entry.depth) || 0),
    pv: Array.isArray(entry.pv) ? entry.pv.filter(Number.isInteger).slice(0, 24) : [],
    nodes: Math.max(0, Number(entry.nodes) || 0),
    updatedAt: Math.max(0, Number(entry.updatedAt) || 0)
  };
}

function shouldReplacePersistedAnalysis(existing, next){
  if (!existing) return true;
  if (next.depth !== existing.depth) return next.depth > existing.depth;
  if (isForcedScore(next.raw) !== isForcedScore(existing.raw)) return isForcedScore(next.raw);
  if ((next.pv?.length || 0) !== (existing.pv?.length || 0)) return (next.pv?.length || 0) > (existing.pv?.length || 0);
  if (Math.abs(next.raw - existing.raw) > 0.5) return Math.abs(next.raw) > Math.abs(existing.raw);
  if (next.nodes > existing.nodes) {
    return next.nodes >= existing.nodes * 1.25 && (next.nodes - existing.nodes) >= 1e9;
  }
  return false;
}

function openAnalysisDb(){
  if (analysisDbPromise !== null) return analysisDbPromise;
  if (typeof indexedDB === 'undefined') {
    analysisDbPromise = Promise.resolve(null);
    return analysisDbPromise;
  }

  analysisDbPromise = new Promise((resolve)=>{
    const request = indexedDB.open(ANALYSIS_DB_NAME, ANALYSIS_DB_VERSION);

    request.onupgradeneeded = ()=>{
      const db = request.result;
      if (!db.objectStoreNames.contains(ANALYSIS_STORE_NAME)) {
        db.createObjectStore(ANALYSIS_STORE_NAME, { keyPath: 'key' });
      }
    };

    request.onsuccess = ()=> resolve(request.result);
    request.onerror = ()=>{
      console.warn('Analysis cache unavailable:', request.error);
      resolve(null);
    };
  });

  return analysisDbPromise;
}

async function readPersistedAnalysis(key){
  if (analysisMemory.has(key)) return analysisMemory.get(key);
  const db = await openAnalysisDb();
  if (!db) return null;

  return new Promise((resolve)=>{
    const tx = db.transaction(ANALYSIS_STORE_NAME, 'readonly');
    const req = tx.objectStore(ANALYSIS_STORE_NAME).get(key);
    req.onsuccess = ()=>{
      const normalized = normalizePersistedAnalysis(req.result);
      if (normalized) analysisMemory.set(key, normalized);
      resolve(normalized);
    };
    req.onerror = ()=> resolve(null);
  });
}

async function persistAnalysisEntry(entry){
  const normalized = normalizePersistedAnalysis(entry);
  if (!normalized) return null;

  const existing = analysisMemory.get(normalized.key);
  if (!shouldReplacePersistedAnalysis(existing, normalized)) return existing;

  analysisMemory.set(normalized.key, normalized);
  const db = await openAnalysisDb();
  if (!db) return normalized;

  return new Promise((resolve)=>{
    const tx = db.transaction(ANALYSIS_STORE_NAME, 'readwrite');
    tx.objectStore(ANALYSIS_STORE_NAME).put(normalized);
    tx.oncomplete = ()=> resolve(normalized);
    tx.onerror = ()=> resolve(normalized);
  });
}

function getKnownAnalysisSync(bd, toPlay){
  return analysisMemory.get(getAnalysisPositionKey(bd, toPlay)) || null;
}

function convertStoredAnalysisToPlayerPerspective(entry, player){
  if (!entry) return null;
  return {
    score: player === 1 ? entry.raw : -entry.raw,
    pv: entry.pv.slice(),
    depth: entry.depth,
    nodes: entry.nodes,
    source: 'cache'
  };
}

async function restorePersistedAnalysisForCurrentPosition(jobId, bd, toPlay){
  const key = getAnalysisPositionKey(bd, toPlay);
  const entry = await readPersistedAnalysis(key);
  if (!entry) return;
  if (jobId !== analysisJobId || gameOver) return;
  if (getAnalysisPositionKey(board, currentPlayer) !== key) return;
  if (latestAnalysis && latestAnalysis.depth >= entry.depth) return;

  latestAnalysis = {
    jobId,
    raw: entry.raw,
    depth: entry.depth,
    pv: entry.pv.slice(),
    nps: ratingToRuminationNps(aiRatingInput?.value || 1500),
    searchingDepth: entry.depth,
    nodes: entry.nodes,
    positionKey: key,
    persistent: true
  };

  renderEvalFromRaw(entry.raw, {
    nodes: entry.nodes,
    depth: entry.depth,
    live: true,
    pv: entry.pv,
    nps: ratingToRuminationNps(aiRatingInput?.value || 1500),
    searchingDepth: entry.depth
  });
}

function persistCurrentAnalysisSnapshot(bd, toPlay, analysis){
  if (!analysis || !analysis.depth || !analysis.pv?.length) return;
  const key = getAnalysisPositionKey(bd, toPlay);
  void persistAnalysisEntry({
    key,
    toPlay,
    raw: analysis.raw,
    depth: analysis.depth,
    pv: analysis.pv.slice(0, 24),
    nodes: analysis.nodes || 0,
    updatedAt: Date.now()
  });
}

function getMovePriorityForSearch(bd, toPlay, col, pvMove=null){
  const opponent = toPlay === 1 ? 2 : 1;
  const res = applyMove(bd, col, toPlay);
  if (!res) return -SEARCH_INF;
  if (localCheckWin(res.board, res.row, col, toPlay)) return 10000000;
  const ownThreats = findImmediateWinningColsFor(res.board, toPlay).length;
  const oppThreats = findImmediateWinningColsFor(res.board, opponent).length;
  const signedPattern = (toPlay === 1 ? 1 : -1) * computePatternScore(res.board);
  let priority = signedPattern + (ownThreats * 4000) - (oppThreats * 4500) - (Math.abs(3 - col) * 25);
  if (pvMove === col) priority += 9000000;
  return priority;
}

function getOrderedMovesForSearch(bd, toPlay, pvMove=null){
  const valid = getValidMoves(bd);
  valid.sort((a, b)=>{
    const diff = getMovePriorityForSearch(bd, toPlay, b, pvMove) - getMovePriorityForSearch(bd, toPlay, a, pvMove);
    if (diff !== 0) return diff;
    return SEARCH_MOVE_ORDER.indexOf(a) - SEARCH_MOVE_ORDER.indexOf(b);
  });
  return valid;
}

function searchNegamax(bd, toPlay, depth, alpha, beta, ply, state){
  if (performance.now() >= state.deadline) throw state.timeout;

  state.nodes += 1;
  const valid = getValidMoves(bd);
  if (valid.length === 0) return { score: 0, pv: [] };

  const key = boardKeyForSearch(bd, toPlay);
  const cached = state.tt.get(key);
  const alphaOrig = alpha;
  const betaOrig = beta;

  if (cached && cached.depth >= depth) {
    if (cached.flag === SEARCH_TT_EXACT) return { score: cached.score, pv: cached.bestMove === null ? [] : [cached.bestMove] };
    if (cached.flag === SEARCH_TT_LOWER) alpha = Math.max(alpha, cached.score);
    else if (cached.flag === SEARCH_TT_UPPER) beta = Math.min(beta, cached.score);
    if (alpha >= beta) return { score: cached.score, pv: cached.bestMove === null ? [] : [cached.bestMove] };
  }

  if (depth === 0) return { score: staticSearchEval(bd, toPlay), pv: [] };

  let bestScore = -SEARCH_INF;
  let bestMove = null;
  let bestPv = [];
  const opponent = toPlay === 1 ? 2 : 1;
  const orderedMoves = getOrderedMovesForSearch(bd, toPlay, cached ? cached.bestMove : null);

  for (const col of orderedMoves) {
    const res = applyMove(bd, col, toPlay);
    if (!res) continue;

    let score;
    let childPv = [];
    if (localCheckWin(res.board, res.row, col, toPlay)) {
      score = FORCED_WIN_SCORE - (ply + 1);
    } else {
      const child = searchNegamax(res.board, opponent, depth - 1, -beta, -alpha, ply + 1, state);
      score = -child.score;
      childPv = child.pv;
    }

    if (score > bestScore) {
      bestScore = score;
      bestMove = col;
      bestPv = [col, ...childPv];
    }

    alpha = Math.max(alpha, bestScore);
    if (alpha >= beta) break;
  }

  let flag = SEARCH_TT_EXACT;
  if (bestScore <= alphaOrig) flag = SEARCH_TT_UPPER;
  else if (bestScore >= betaOrig) flag = SEARCH_TT_LOWER;

  state.tt.set(key, { depth, score: bestScore, bestMove, flag });
  return { score: bestScore, pv: bestPv };
}

function ratingToReviewDepth(rating, empties){
  if (rating < 300) return Math.min(empties, 4);
  if (rating < 600) return Math.min(empties, 5);
  if (rating < 900) return Math.min(empties, 6);
  if (rating < 1200) return Math.min(empties, 7);
  if (rating < 1500) return Math.min(empties, 8);
  if (rating < 1800) return Math.min(empties, 9);
  if (rating < 2100) return Math.min(empties, 10);
  return Math.min(empties, 11);
}

function ratingToReviewBudgetMs(rating, empties){
  if (empties > 30) return rating < 1500 ? 6 : 8;
  if (empties > 20) return rating < 1500 ? 8 : 12;
  if (empties > 10) return rating < 1500 ? 10 : 16;
  return rating < 1500 ? 12 : 20;
}

function analyzePositionSync(bd, toPlay, rating, options={}){
  const empties = MAX_GAME_PLIES - countFilledCells(bd);
  const maxDepth = Math.max(1, Math.min(options.maxDepth ?? ratingToReviewDepth(rating, empties), empties));
  const deadline = performance.now() + Math.max(1, options.budgetMs ?? ratingToReviewBudgetMs(rating, empties));
  const timeout = { timeout: true };
  const state = {
    tt: new Map(),
    nodes: 0,
    deadline,
    timeout,
    best: { score: staticSearchEval(bd, toPlay), pv: [], depth: 0 }
  };

  for (let depth = 1; depth <= maxDepth; depth++) {
    try {
      const result = searchNegamax(bd, toPlay, depth, -SEARCH_INF, SEARCH_INF, 0, state);
      state.best = { score: result.score, pv: result.pv, depth };
      if (isForcedScore(result.score)) break;
    } catch (error) {
      if (error !== timeout && !(error && error.timeout)) throw error;
      break;
    }
  }

  return {
    score: state.best.score,
    pv: state.best.pv.slice(),
    depth: state.best.depth,
    nodes: state.nodes
  };
}

function rawToNormalized(raw){
  return Math.max(-100, Math.min(100, Math.round(100 * Math.tanh(raw / 2600))));
}

function isForcedScore(raw){
  return Number.isFinite(raw) && Math.abs(Math.trunc(raw)) >= FORCED_WIN_SCORE - MAX_GAME_PLIES;
}

function forcedScoreToPlies(raw){
  return Math.max(1, FORCED_WIN_SCORE - Math.abs(Math.trunc(raw)));
}

function getEvalPhrase(raw, norm){
  if (isForcedScore(raw)) {
    const plies = forcedScoreToPlies(raw);
    return raw > 0 ? `Cyan wins in ${plies}` : `Red wins in ${plies}`;
  }
  if (norm === 0) return 'Equal position';
  return norm > 0 ? 'Cyan advantage' : 'Red advantage';
}

function isCurrentTurnControlledByAI(){
  return !gameOver && Boolean(aiToggle?.checked) && Number(aiPlayerSelect?.value) === currentPlayer;
}

function interpolateLogRate(rating, left, right){
  if (left.rate <= 0 || right.rate <= 0 || right.rating === left.rating) return left.rate;
  const t = (rating - left.rating) / (right.rating - left.rating);
  return Math.exp(Math.log(left.rate) + (Math.log(right.rate) - Math.log(left.rate)) * t);
}

function ratingToRuminationNps(rating){
  const anchors = [
    { rating: 250, rate: 5e9 },
    { rating: 500, rate: 10e9 },
    { rating: 750, rate: 20e9 },
    { rating: 1000, rate: 40e9 },
    { rating: 1250, rate: 80e9 },
    { rating: 1500, rate: 100e9 },
    { rating: 1750, rate: 125e9 },
    { rating: 2000, rate: 250e9 },
    { rating: 2250, rate: 500e9 },
    { rating: 2500, rate: 1e12 },
    { rating: 2750, rate: 2e12 }
  ];

  const numeric = Math.max(1, Number(rating) || 1500);
  if (numeric <= anchors[0].rating) return interpolateLogRate(numeric, anchors[0], anchors[1]);
  for (let i = 1; i < anchors.length; i++) {
    if (numeric <= anchors[i].rating) return interpolateLogRate(numeric, anchors[i - 1], anchors[i]);
  }
  return interpolateLogRate(numeric, anchors[anchors.length - 2], anchors[anchors.length - 1]);
}

function trimFixed(value, decimals){
  return value.toFixed(decimals).replace(/\.?0+$/,'');
}

function formatLargeQuantity(value, suffix){
  const abs = Math.abs(value);
  const units = [
    { threshold: 1e12, label: 'T' },
    { threshold: 1e9, label: 'B' },
    { threshold: 1e6, label: 'M' },
    { threshold: 1e3, label: 'k' }
  ];

  for (const unit of units) {
    if (abs >= unit.threshold) {
      const scaled = value / unit.threshold;
      const decimals = Math.abs(scaled) >= 100 ? 0 : Math.abs(scaled) >= 10 ? 1 : 2;
      return `${trimFixed(scaled, decimals)}${unit.label} ${suffix}`;
    }
  }

  return `${Math.round(value)} ${suffix}`;
}

function formatNodeRate(nps){
  return formatLargeQuantity(nps, 'n/s');
}

function formatNodeCount(nodes){
  return formatLargeQuantity(nodes, 'nodes');
}

function formatPrincipalVariation(pv){
  if (!pv || !pv.length) return '';
  return `Top line: ${pv.map((col)=>col + 1).join(' - ')}`;
}

function renderEvalFromRaw(raw, {nodes=0, depth=0, live=false, pv=[], nps=0, searchingDepth=0}={}) {
  const norm = rawToNormalized(raw);
  const ruminationNps = ratingToRuminationNps(aiRatingInput?.value || 1500);

  const cyanPct = (norm + 100) / 2;
  const redPct = 100 - cyanPct;

  evalLeft.style.width = cyanPct + "%";
  evalRight.style.width = redPct + "%";
  evalBar?.classList.toggle('analyzing', live && !gameOver);

  const details = [];
  if (isCurrentTurnControlledByAI()) details.push('AI thinking');
  if (depth > 0) details.push(`depth ${depth}`);
  else if (searchingDepth > 0) details.push(`searching d${searchingDepth}`);
  if (nodes > 0) details.push(formatNodeCount(nodes));
  if (ruminationNps > 0) details.push(`rumination ${formatNodeRate(ruminationNps)}`);
  if (live && depth === 0 && searchingDepth === 0) details.push('live');

  const detailText = details.length ? ` · ${details.join(' · ')}` : '';
  evalLabel.textContent = `Eval: ${norm > 0 ? '+' : ''}${norm} (${getEvalPhrase(raw, norm)}${detailText})`;
  if (evalMeta) evalMeta.textContent = formatPrincipalVariation(pv);
}

function updateThreatsDisplay(){
  const threatsA = findImmediateWinningColsFor(board, 1);
  const threatsB = findImmediateWinningColsFor(board, 2);
  threatsEl.textContent =
    (threatsA.length || threatsB.length)
      ? `A: [${threatsA.map(x=>x+1).join(",")||"-"}] | B: [${threatsB.map(x=>x+1).join(",")||"-"}]`
      : "No immediate winning threats";
}

function createLiveAnalysisWorker(){
  if (typeof Worker === 'undefined') return null;

  const source = `
    const ROWS = ${ROWS};
    const COLS = ${COLS};
    const MAX_GAME_PLIES = ${MAX_GAME_PLIES};
    const FORCED_WIN_SCORE = ${FORCED_WIN_SCORE};
    const SEARCH_EVAL_CAP = ${SEARCH_EVAL_CAP};
    const SEARCH_INF = ${SEARCH_INF};
    ${cloneBoard.toString()}
    ${inBounds.toString()}
    ${getValidMoves.toString()}
    ${applyMove.toString()}
    ${localCheckWin.toString()}
    ${findImmediateWinningColsFor.toString()}
    ${computePatternScore.toString()}

    const MOVE_ORDER = [3,2,4,1,5,0,6];
    const TT_EXACT = 0;
    const TT_LOWER = 1;
    const TT_UPPER = 2;
    const TIMEOUT = { timeout: true };

    function clamp(value, min, max){ return Math.max(min, Math.min(max, value)); }
    function interpolateLogRate(rating, left, right){
      if (left.rate <= 0 || right.rate <= 0 || right.rating === left.rating) return left.rate;
      const t = (rating - left.rating) / (right.rating - left.rating);
      return Math.exp(Math.log(left.rate) + (Math.log(right.rate) - Math.log(left.rate)) * t);
    }
    function ratingToRuminationNps(rating){
      const anchors = [
        { rating: 250, rate: 5e9 },
        { rating: 500, rate: 10e9 },
        { rating: 750, rate: 20e9 },
        { rating: 1000, rate: 40e9 },
        { rating: 1250, rate: 80e9 },
        { rating: 1500, rate: 100e9 },
        { rating: 1750, rate: 125e9 },
        { rating: 2000, rate: 250e9 },
        { rating: 2250, rate: 500e9 },
        { rating: 2500, rate: 1e12 },
        { rating: 2750, rate: 2e12 }
      ];
      const clamped = Math.max(1, Number(rating) || 1500);
      if (clamped <= anchors[0].rating) return interpolateLogRate(clamped, anchors[0], anchors[1]);
      for (let i = 1; i < anchors.length; i++) {
        if (clamped <= anchors[i].rating) return interpolateLogRate(clamped, anchors[i - 1], anchors[i]);
      }
      return interpolateLogRate(clamped, anchors[anchors.length - 2], anchors[anchors.length - 1]);
    }
    function toCyanScore(score, rootPlayer){ return rootPlayer === 1 ? score : -score; }
    function isForcedScore(raw){ return Number.isFinite(raw) && Math.abs(Math.trunc(raw)) >= FORCED_WIN_SCORE - MAX_GAME_PLIES; }

    function staticSearchEval(bd, toPlay){
      const opponent = toPlay === 1 ? 2 : 1;
      const ownWins = findImmediateWinningColsFor(bd, toPlay).length;
      const oppWins = findImmediateWinningColsFor(bd, opponent).length;
      const signedPattern = (toPlay === 1 ? 1 : -1) * computePatternScore(bd);
      return clamp(signedPattern + (ownWins * 2200) - (oppWins * 2600), -SEARCH_EVAL_CAP, SEARCH_EVAL_CAP);
    }

    function boardKey(bd, toPlay){
      let key = '' + toPlay;
      for(let r=0;r<ROWS;r++) for(let c=0;c<COLS;c++) key += bd[r][c];
      return key;
    }

    function getMovePriority(bd, toPlay, col, pvMove){
      const opponent = toPlay === 1 ? 2 : 1;
      const res = applyMove(bd, col, toPlay);
      if (!res) return -SEARCH_INF;
      if (localCheckWin(res.board, res.row, col, toPlay)) return 10000000;
      const ownThreats = findImmediateWinningColsFor(res.board, toPlay).length;
      const oppThreats = findImmediateWinningColsFor(res.board, opponent).length;
      const signedPattern = (toPlay === 1 ? 1 : -1) * computePatternScore(res.board);
      let priority = signedPattern + (ownThreats * 4000) - (oppThreats * 4500) - (Math.abs(3 - col) * 25);
      if (pvMove === col) priority += 9000000;
      return priority;
    }

    function getOrderedMoves(bd, toPlay, pvMove){
      const valid = getValidMoves(bd);
      valid.sort((a, b)=>{
        const diff = getMovePriority(bd, toPlay, b, pvMove) - getMovePriority(bd, toPlay, a, pvMove);
        if (diff !== 0) return diff;
        return MOVE_ORDER.indexOf(a) - MOVE_ORDER.indexOf(b);
      });
      return valid;
    }

    function negamax(bd, toPlay, depth, alpha, beta, ply, state){
      if (state.jobId !== activeJobId) throw TIMEOUT;
      if (performance.now() >= state.deadline) throw TIMEOUT;

      state.nodes += 1;
      const valid = getValidMoves(bd);
      if (valid.length === 0) return { score: 0, pv: [] };

      const key = boardKey(bd, toPlay);
      const cached = state.tt.get(key);
      const alphaOrig = alpha;
      const betaOrig = beta;

      if (cached && cached.depth >= depth) {
        if (cached.flag === TT_EXACT) {
          return { score: cached.score, pv: cached.bestMove === null ? [] : [cached.bestMove] };
        }
        if (cached.flag === TT_LOWER) alpha = Math.max(alpha, cached.score);
        else if (cached.flag === TT_UPPER) beta = Math.min(beta, cached.score);
        if (alpha >= beta) {
          return { score: cached.score, pv: cached.bestMove === null ? [] : [cached.bestMove] };
        }
      }

      if (depth === 0) return { score: staticSearchEval(bd, toPlay), pv: [] };

      let bestScore = -SEARCH_INF;
      let bestMove = null;
      let bestPv = [];
      const opponent = toPlay === 1 ? 2 : 1;
      const orderedMoves = getOrderedMoves(bd, toPlay, cached ? cached.bestMove : null);

      for (const col of orderedMoves) {
        const res = applyMove(bd, col, toPlay);
        if (!res) continue;

        let score;
        let childPv = [];

        if (localCheckWin(res.board, res.row, col, toPlay)) {
          score = FORCED_WIN_SCORE - (ply + 1);
        } else {
          const child = negamax(res.board, opponent, depth - 1, -beta, -alpha, ply + 1, state);
          score = -child.score;
          childPv = child.pv;
        }

        if (score > bestScore) {
          bestScore = score;
          bestMove = col;
          bestPv = [col, ...childPv];
        }

        alpha = Math.max(alpha, bestScore);
        if (alpha >= beta) break;
      }

      let flag = TT_EXACT;
      if (bestScore <= alphaOrig) flag = TT_UPPER;
      else if (bestScore >= betaOrig) flag = TT_LOWER;

      state.tt.set(key, { depth, score: bestScore, bestMove, flag });
      return { score: bestScore, pv: bestPv };
    }

    function ratingToAnalysisMaxDepth(rating, empties){
      if (rating < 300) return Math.min(empties, 6);
      if (rating < 600) return Math.min(empties, 7);
      if (rating < 900) return Math.min(empties, 8);
      if (rating < 1200) return Math.min(empties, 10);
      if (rating < 1500) return Math.min(empties, 12);
      if (rating < 1800) return Math.min(empties, 14);
      if (rating < 2100) return Math.min(empties, 16);
      if (rating < 2400) return Math.min(empties, 18);
      if (rating < 2700) return Math.min(empties, 20);
      return Math.min(empties, 22);
    }

    function ratingToBurstMs(rating){
      if (rating < 600) return 5;
      if (rating < 1200) return 7;
      if (rating < 1800) return 9;
      if (rating < 2400) return 11;
      return 13;
    }

    function postProgress(state){
      const elapsedMs = Math.max(1, performance.now() - state.startedAt);
      self.postMessage({
        type: 'progress',
        jobId: state.jobId,
        raw: toCyanScore(state.best.score, state.rootPlayer),
        depth: state.best.depth,
        searchingDepth: state.searchingDepth,
        nodes: Math.round((state.ruminationNps * elapsedMs) / 1000),
        nps: Math.round((state.nodes * 1000) / elapsedMs),
        pv: state.best.pv,
        live: true,
        exact: isForcedScore(toCyanScore(state.best.score, state.rootPlayer))
      });
    }

    let activeJobId = 0;

    function runIterative(state){
      if (state.jobId !== activeJobId) return;
      state.deadline = performance.now() + state.burstMs;

      while (state.jobId === activeJobId && performance.now() < state.deadline && state.nextDepth <= state.maxDepth) {
        state.searchingDepth = state.nextDepth;
        try {
          const result = negamax(state.board, state.toPlay, state.nextDepth, -SEARCH_INF, SEARCH_INF, 0, state);
          state.best = { score: result.score, pv: result.pv, depth: state.nextDepth };
          state.nextDepth += 1;
          postProgress(state);
          if (isForcedScore(toCyanScore(result.score, state.rootPlayer))) return;
        } catch (error) {
          if (error !== TIMEOUT && !(error && error.timeout)) throw error;
          break;
        }
      }

      if (state.nextDepth > state.maxDepth) {
        state.searchingDepth = state.maxDepth;
      }
      postProgress(state);
      if (state.jobId !== activeJobId) return;
      setTimeout(() => runIterative(state), state.cooldownMs);
    }

    self.onmessage = (event) => {
      const data = event.data || {};
      if (data.type === 'stop') {
        activeJobId = data.jobId || (activeJobId + 1);
        return;
      }
      if (data.type !== 'start') return;

      activeJobId = data.jobId;
      const filled = data.board.reduce((sum, row) => sum + row.filter(Boolean).length, 0);
      const empties = MAX_GAME_PLIES - filled;
      const state = {
        jobId: data.jobId,
        board: cloneBoard(data.board),
        toPlay: data.currentPlayer,
        rootPlayer: data.currentPlayer,
        startedAt: performance.now(),
        nodes: 0,
        ruminationNps: ratingToRuminationNps(data.rating),
        tt: new Map(),
        best: { score: staticSearchEval(data.board, data.currentPlayer), pv: [], depth: 0 },
        searchingDepth: 1,
        nextDepth: 1,
        maxDepth: ratingToAnalysisMaxDepth(data.rating, empties),
        burstMs: ratingToBurstMs(data.rating),
        cooldownMs: 24,
        deadline: 0
      };

      postProgress(state);
      runIterative(state);
    };
  `;

  const blob = new Blob([source], { type: 'text/javascript' });
  const url = URL.createObjectURL(blob);
  const worker = new Worker(url);
  URL.revokeObjectURL(url);
  return worker;
}

function ensureAnalysisWorker(){
  if (analysisWorker) return analysisWorker;
  try {
    analysisWorker = createLiveAnalysisWorker();
  } catch (error) {
    console.warn('Live analysis worker unavailable:', error);
    analysisWorker = null;
  }
  if (!analysisWorker) return null;

  analysisWorker.onmessage = (event)=>{
    const data = event.data || {};
    if (data.type !== 'progress' || data.jobId !== analysisJobId || gameOver) return;
    const positionKey = getAnalysisPositionKey(board, currentPlayer);
    if (latestAnalysis?.positionKey === positionKey && latestAnalysis.depth > data.depth) {
      latestAnalysis.nodes = Math.max(latestAnalysis.nodes || 0, data.nodes || 0);
      renderEvalFromRaw(latestAnalysis.raw, {
        nodes: latestAnalysis.nodes,
        depth: latestAnalysis.depth,
        live: data.live,
        pv: latestAnalysis.pv,
        nps: data.nps,
        searchingDepth: Math.max(data.searchingDepth || 0, latestAnalysis.depth)
      });
      return;
    }
    latestAnalysis = {
      jobId: data.jobId,
      raw: data.raw,
      depth: data.depth,
      pv: Array.isArray(data.pv) ? data.pv.slice() : [],
      nodes: data.nodes,
      nps: data.nps,
      searchingDepth: data.searchingDepth,
      positionKey
    };
    persistCurrentAnalysisSnapshot(board, currentPlayer, latestAnalysis);
    renderEvalFromRaw(data.raw, {
      nodes: data.nodes,
      depth: data.depth,
      live: data.live,
      pv: data.pv,
      nps: data.nps,
      searchingDepth: data.searchingDepth
    });
  };
  analysisWorker.onerror = (event)=>{
    console.error('Live analysis worker error:', event.message || event);
    analysisWorker?.terminate();
    analysisWorker = null;
    if (!gameOver) {
      const rating = Number(aiRatingInput.value || 1500);
      renderEvalFromRaw(evaluateBoardForCyan(board, currentPlayer, rating), { live: false });
    }
  };

  return analysisWorker;
}

function stopLiveAnalysis(){
  analysisJobId += 1;
  latestAnalysis = null;
  if (analysisWorker) analysisWorker.postMessage({ type: 'stop', jobId: analysisJobId });
  evalBar?.classList.remove('analyzing');
  if (evalMeta) evalMeta.textContent = '';
}

function startLiveAnalysis(){
  stopLiveAnalysis();

  const quickRaw = computeImmediateTurnRaw(board, currentPlayer);
  const baselineRaw = quickRaw ?? Math.max(-1600, Math.min(1600, computePatternScore(board)));
  renderEvalFromRaw(baselineRaw, { live: quickRaw === null });

  const worker = ensureAnalysisWorker();
  if (!worker) {
    const rating = Number(aiRatingInput.value || 1500);
    renderEvalFromRaw(evaluateBoardForCyan(board, currentPlayer, rating), { live: false });
    return;
  }

  const rating = Number(aiRatingInput.value || 1500);
  const jobId = ++analysisJobId;
  worker.postMessage({
    type: 'start',
    jobId,
    board: cloneBoard(board),
    currentPlayer,
    rating
  });
  void restorePersistedAnalysisForCurrentPosition(jobId, cloneBoard(board), currentPlayer);
}

function formatReviewScore(raw){
  if (raw === null || raw === undefined || !Number.isFinite(raw)) return '?';
  if (isForcedScore(raw)) return `${raw > 0 ? 'W' : 'L'}${forcedScoreToPlies(raw)}`;
  return `${raw > 0 ? '+' : ''}${(raw / 1000).toFixed(2)}`;
}

function formatReviewLoss(loss, bestRaw, playedRaw){
  if (loss <= 0) return 'No loss';
  if (isForcedScore(bestRaw) && bestRaw > 0 && (!isForcedScore(playedRaw) || playedRaw <= 0)) return 'Missed forced win';
  if (!isForcedScore(bestRaw) && isForcedScore(playedRaw) && playedRaw < 0) return 'Dropped into forced loss';
  if (isForcedScore(bestRaw) && isForcedScore(playedRaw) && bestRaw > 0 && playedRaw > 0) {
    return `Win slowed by ${Math.max(0, forcedScoreToPlies(playedRaw) - forcedScoreToPlies(bestRaw))} ply`;
  }
  return `${(loss / 1000).toFixed(2)} loss`;
}

function getReviewReferenceAnalysis(boardBefore, player, rating){
  const positionKey = getAnalysisPositionKey(boardBefore, player);
  if (latestAnalysis?.jobId === analysisJobId && latestAnalysis.positionKey === positionKey && latestAnalysis.depth >= 2 && latestAnalysis.pv?.length) {
    return {
      score: player === 1 ? latestAnalysis.raw : -latestAnalysis.raw,
      pv: latestAnalysis.pv.slice(),
      depth: latestAnalysis.depth,
      nodes: latestAnalysis.nodes || 0,
      source: 'live'
    };
  }
  const known = convertStoredAnalysisToPlayerPerspective(getKnownAnalysisSync(boardBefore, player), player);
  if (known && known.depth >= 2 && known.pv?.length) return known;
  return analyzePositionSync(boardBefore, player, rating);
}

function findBestThreatReductionCols(bd, player){
  const opponent = player === 1 ? 2 : 1;
  const valid = getValidMoves(bd);
  const scored = valid.map((col)=>{
    const res = applyMove(bd, col, player);
    if (!res) return { col, opponentThreats: Infinity };
    return { col, opponentThreats: findImmediateWinningColsFor(res.board, opponent).length };
  });
  const bestThreatCount = Math.min(...scored.map((entry)=>entry.opponentThreats));
  return scored.filter((entry)=>entry.opponentThreats === bestThreatCount).map((entry)=>entry.col);
}

function classifyReviewedMove({col, bestMove, bestRaw, playedRaw, scoreLoss, playerWinsBefore, bestDefenses, opponentThreatsBefore}){
  if (playerWinsBefore.includes(col)) return { tag: 'Winning Move', reason: 'Finished the game immediately.' };
  if (playerWinsBefore.length && !playerWinsBefore.includes(col)) {
    return { tag: 'Clown', reason: `Missed a winning move in column ${playerWinsBefore[0] + 1}.` };
  }
  if (opponentThreatsBefore.length && !bestDefenses.includes(col)) {
    return { tag: 'Clown', reason: `Ignored the critical defense. Best defense was column ${bestDefenses[0] + 1}.` };
  }
  if (isForcedScore(bestRaw) && bestRaw > 0 && (!isForcedScore(playedRaw) || playedRaw <= 0)) {
    return { tag: 'Clown', reason: `Missed a forced win. Best was column ${bestMove + 1}.` };
  }
  if (!isForcedScore(bestRaw) && isForcedScore(playedRaw) && playedRaw < 0) {
    return { tag: 'Clown', reason: 'Turned a playable position into a forced loss.' };
  }
  if (bestMove === col && scoreLoss <= 40) {
    return { tag: 'Sigma', reason: 'Matched the engine top move.' };
  }
  if (scoreLoss <= 120) {
    return { tag: 'Chad', reason: bestMove === col ? 'Near-perfect move.' : `Almost as strong as the top move in column ${bestMove + 1}.` };
  }
  if (scoreLoss <= 400) {
    return { tag: 'Good', reason: `Solid move, but column ${bestMove + 1} was cleaner.` };
  }
  if (scoreLoss <= 1000) {
    return { tag: 'Ok', reason: `Playable, but it gave away some equity versus column ${bestMove + 1}.` };
  }
  if (scoreLoss <= 2200) {
    return { tag: 'Strange', reason: `This drifted away from the best plan in column ${bestMove + 1}.` };
  }
  if (scoreLoss <= 4200) {
    return { tag: 'Bad', reason: `A serious inaccuracy. Best move was column ${bestMove + 1}.` };
  }
  return { tag: 'Clown', reason: `Major blunder. Best move was column ${bestMove + 1}.` };
}

function analyzeMoveForReview(boardBefore, col, player, rating){
  const opponent = player === 1 ? 2 : 1;
  const beforeAnalysis = getReviewReferenceAnalysis(boardBefore, player, rating);
  const bestMove = beforeAnalysis.pv?.[0] ?? col;
  const playerWinsBefore = findImmediateWinningColsFor(boardBefore, player);
  const opponentThreatsBefore = findImmediateWinningColsFor(boardBefore, opponent);
  const bestDefenses = opponentThreatsBefore.length ? findBestThreatReductionCols(boardBefore, player) : [];

  const played = applyMove(boardBefore, col, player);
  if (!played) {
    return {
      player,
      col,
      beforeRaw: beforeAnalysis.score,
      afterRaw: beforeAnalysis.score,
      bestRaw: beforeAnalysis.score,
      bestMove,
      bestPv: beforeAnalysis.pv || [],
      playedPv: [],
      scoreLoss: 0,
      depth: beforeAnalysis.depth || 0,
      tag: 'Ok',
      reason: 'Illegal move was ignored.'
    };
  }

  let afterRaw;
  let afterPv = [];
  let afterDepth = 0;

  if (localCheckWin(played.board, played.row, col, player)) {
    afterRaw = FORCED_WIN_SCORE - 1;
  } else if (getValidMoves(played.board).length === 0) {
    afterRaw = 0;
  } else if (bestMove === col && beforeAnalysis.depth >= 2) {
    afterRaw = beforeAnalysis.score;
    afterPv = beforeAnalysis.pv.slice(1);
    afterDepth = Math.max(0, beforeAnalysis.depth - 1);
  } else {
    const replyAnalysis = analyzePositionSync(played.board, opponent, rating, { maxDepth: Math.max(1, (beforeAnalysis.depth || ratingToReviewDepth(rating, MAX_GAME_PLIES - countFilledCells(played.board))) - 1) });
    afterRaw = -replyAnalysis.score;
    afterPv = replyAnalysis.pv.slice();
    afterDepth = replyAnalysis.depth;
  }

  const bestRaw = beforeAnalysis.score;
  const scoreLoss = Math.max(0, bestRaw - afterRaw);
  const classification = classifyReviewedMove({
    col,
    bestMove,
    bestRaw,
    playedRaw: afterRaw,
    scoreLoss,
    playerWinsBefore,
    bestDefenses,
    opponentThreatsBefore
  });

  return {
    player,
    col,
    beforeRaw: bestRaw,
    afterRaw,
    bestRaw,
    bestMove,
    bestPv: beforeAnalysis.pv || [],
    playedPv: afterPv,
    scoreLoss,
    depth: Math.max(beforeAnalysis.depth || 0, afterDepth || 0),
    tag: classification.tag,
    reason: classification.reason,
    opponentThreatsBefore,
    playerWinsBefore,
    lossText: formatReviewLoss(scoreLoss, bestRaw, afterRaw)
  };
}

function updateEvalAndThreats() {

    if (gameOver) {
        stopLiveAnalysis();
        let norm = 0;
        let winnerText = "Draw";

        if (lastWinningCells && lastWinningCells.length > 0) {
            const [r, c] = lastWinningCells[0];
            if (board[r][c] === 1) { norm = 100; winnerText = "Player A wins!"; }
            else if (board[r][c] === 2) { norm = -100; winnerText = "Player B wins!"; }
        }

        const cyanPct = (norm + 100) / 2;
        const redPct = 100 - cyanPct;

        evalLeft.style.width = cyanPct + "%";
        evalRight.style.width = redPct + "%";
        evalLabel.textContent = `Eval: ${norm} (${winnerText})`;
        threatsEl.textContent = "Game Over";
        return;
    }

    updateThreatsDisplay();
    startLiveAnalysis();
}
function formatHumanEval(){ const rating = Number(aiRatingInput.value || 1500); const raw = evaluateBoardForCyan(board, currentPlayer, rating); return ((raw/1000)>=0?'+':'') + (raw/1000).toFixed(2); }

// =================== Move handling ===================
function isBoardFull(bd=board){ return bd.every(row => row.every(cell => cell!==0)); }

function handleColumnClick(col, byAI=false){
  if(gameOver) return;
  if(!byAI && aiToggle?.checked && Number(aiPlayerSelect?.value) === currentPlayer) return;

  // find row
  let targetRow = -1;
  for(let r=ROWS-1;r>=0;r--) if(board[r][col]===0){ targetRow=r; break; }
  if(targetRow === -1) return;

  const boardBefore = cloneBoard(board);
  const reviewEntry = analyzeMoveForReview(boardBefore, col, currentPlayer, Number(aiRatingInput.value || 1500));

  board[targetRow][col] = currentPlayer;
  const cellIndex = targetRow * COLS + col;
  const cell = boardEl.children[cellIndex];
  const disc = cell?.querySelector('.disc');
  if(disc){
    disc.classList.remove('empty');
    disc.classList.add(currentPlayer===1?'playera':'playerb');
    disc.classList.add('drop');
    setTimeout(()=>disc.classList.remove('drop'), 380);
  }

  if(checkWin(targetRow,col,currentPlayer)){
    gameOver = true;
    updateMessage(`Player ${currentPlayer===1?'A':'B'} wins!`);
    highlightWinningLine();
    moveHistory.push(reviewEntry);
    updateEvalAndThreats();
    renderBoard();
    return;
  }

  moveHistory.push(reviewEntry);

  if(isBoardFull()){
    gameOver = true;
    updateMessage('Tie — board is full');
    updateEvalAndThreats();
    return;
  }

  currentPlayer = currentPlayer===1?2:1;
  updateMessage();
  updateEvalAndThreats();

  // AI turn scheduling
  triggerAITurnIfNeeded();
}

function triggerAITurnIfNeeded(){
  // clear any pending AI move
  if(pendingAITimeout) {
    clearTimeout(pendingAITimeout);
    pendingAITimeout = null;
  }
  
  if(!gameOver && aiToggle?.checked && Number(aiPlayerSelect?.value) === currentPlayer){
    pendingAITimeout = setTimeout(()=> {
      pendingAITimeout = null;
      ultraChooseAndPlay();
    }, 220 + Math.random()*240);
  }
}

// =================== Review UI ===================
showReviewBtn?.addEventListener('click', ()=>{
  if(!reviewEl) return;
  reviewEl.hidden = !reviewEl.hidden;
  if(!reviewEl.hidden) renderReview();
});
function renderReview(){
  if(!reviewEl) return;
  reviewEl.innerHTML = '';
  const h = document.createElement('h3'); h.textContent='Game Review'; reviewEl.appendChild(h);
  const summary = document.createElement('div'); summary.className='summary';
  const counts = {Sigma:0,Chad:0,Good:0,Ok:0,Strange:0,Bad:0,Clown:0,'Winning Move':0};
  for(const m of moveHistory) counts[m.tag]=(counts[m.tag]||0)+1;
  summary.textContent = `Sigma:${counts.Sigma} Chad:${counts.Chad} Good:${counts.Good} Ok:${counts.Ok} Strange:${counts.Strange} Bad:${counts.Bad} Clown:${counts.Clown} Winning:${counts['Winning Move']}`;
  reviewEl.appendChild(summary);
  moveHistory.forEach((m,i)=>{
    const block = document.createElement('div'); block.className='move';
    if (m.tag) block.classList.add(m.tag.toLowerCase().replace(/\s+/g,'-'));

    if (m.beforeRaw === undefined) {
      const left = document.createElement('div'); left.textContent=`#${i+1} P${m.player===1?'A':'B'} → Col ${m.col+1}`;
      const right = document.createElement('div'); right.innerHTML=`${m.before} → ${m.after} (${m.deltaPercent>=0?'+':''}${m.deltaPercent}%) <span class="tag ${m.tag.toLowerCase().replace(/\s+/g,'-')}">${m.tag}</span>`;
      block.appendChild(left);
      block.appendChild(right);
      reviewEl.appendChild(block);
      return;
    }

    const top = document.createElement('div'); top.className = 'move-main';
    top.innerHTML = `#${i+1} P${m.player===1?'A':'B'} → Col ${m.col+1} <span class="tag ${m.tag.toLowerCase().replace(/\s+/g,'-')}">${m.tag}</span>`;

    const meta = document.createElement('div'); meta.className = 'move-meta';
    meta.textContent = `Best ${Number.isInteger(m.bestMove) ? m.bestMove + 1 : '?'} | ${formatReviewScore(m.beforeRaw)} → ${formatReviewScore(m.afterRaw)} | ${m.lossText}${m.depth ? ` | depth ${m.depth}` : ''}`;

    const line = document.createElement('div'); line.className = 'move-meta';
    line.textContent = m.bestPv?.length ? `Top line: ${m.bestPv.map((step)=>step + 1).join(' - ')}` : 'Top line: unavailable';

    const reason = document.createElement('div'); reason.className = 'move-reason';
    reason.textContent = m.reason || 'No comment.';

    block.appendChild(top);
    block.appendChild(meta);
    block.appendChild(line);
    block.appendChild(reason);
    reviewEl.appendChild(block);
  });
}

// =================== AI: utilities ===================
// Note: orderedMoves removed - MCTS doesn't require static move ordering

// =================== MCTS Node Structure ===================
class MCTSNode {
  constructor(move = null, parent = null, aiPlayer = 1) {
    this.move = move;        // column number or null for root
    this.parent = parent;
    this.children = [];       // array of child nodes
    this.visits = 0;          // N(s)
    this.value = 0;           // cumulative value / reward (from aiPlayer perspective)
    this.aiPlayer = aiPlayer; // which player's turn to move from here
    this.untried = null;      // lazy init: valid moves not yet explored as children
  }

  ucb1() {
    if (this.visits === 0) return Infinity;
    const exploitation = this.value / this.visits;
    const exploration = UCB_EXPLORATION * Math.sqrt(Math.log(this.parent.visits) / this.visits);
    return exploitation + exploration;
  }

  selectChild() {
    // select child with highest UCB1
    return this.children.reduce((best, child) => 
      child.ucb1() > best.ucb1() ? child : best
    );
  }

  expandRandomChild(bd) {
    // get untried moves
    if (this.untried === null) {
      this.untried = getValidMoves(bd);
    }
    if (this.untried.length === 0) return null;
    
    // pick random untried move
    const idx = Math.floor(Math.random() * this.untried.length);
    const col = this.untried[idx];
    this.untried.splice(idx, 1);
    
    // create child node
    const opponent = this.aiPlayer === 1 ? 2 : 1;
    const child = new MCTSNode(col, this, opponent);
    this.children.push(child);
    return child;
  }

  isFullyExpanded(bd) {
    if (this.untried === null) {
      this.untried = getValidMoves(bd);
    }
    return this.untried.length === 0;
  }
}

// =================== MCTS Core Algorithm ===================
function mctsSelection(node, bd) {
  // tree policy: navigate down tree using UCB1
  while (!isTerminal(bd)) {
    if (!node.isFullyExpanded(bd)) {
      return mctsExpansion(node, bd);
    } else if (node.children.length === 0) {
      // terminal state with no moves
      return node;
    } else {
      // select best child by UCB1
      node = node.selectChild();
      // apply move to board
      const res = applyMove(bd, node.move, node.parent.aiPlayer);
      if (!res) return node;
      bd = res.board;
    }
  }
  return node;
}

function mctsExpansion(node, bd) {
  // expand one new child
  const child = node.expandRandomChild(bd);
  if (!child) return node;
  
  const res = applyMove(bd, child.move, node.aiPlayer);
  if (!res) return child;
  
  return child;
}

function mctsSimulation(bd, aiPlayer) {
  // random playout from current board state
  const result = randomPlayout(bd, aiPlayer);
  
  // convert winner to score: +1 if aiPlayer wins, -1 if opponent, 0 draw
  if (result === aiPlayer) return 1;
  if (result === 0) return 0;
  return -1;
}

function mctsBackup(node, reward) {
  // propagate result up the tree
  while (node !== null) {
    node.visits += 1;
    // add reward from node's aiPlayer perspective
    node.value += reward;
    node = node.parent;
  }
}

function isTerminal(bd) {
  // check if board is in terminal state (win/draw)
  const valid = getValidMoves(bd);
  if (valid.length === 0) return true;
  
  // check for any immediate winning moves (quick terminal detection)
  // if any player can win next move, consider it terminal for MCTS purposes
  // (actual win is deeper but we use this as heuristic for playout generation)
  return false;
}

function runMCTS(bd, aiPlayer, numSimulations, rating=1500) {
  const root = new MCTSNode(null, null, aiPlayer);
  root.visits = 1;  // initialize root visits
  
  for (let i = 0; i < numSimulations; i++) {
    let simBd = cloneBoard(bd);
    
    // Selection & Expansion
    let node = mctsSelection(root, simBd);
    
    // Simulation with brute-force evaluation
    const reward = mctsSimulation(simBd, aiPlayer);
    
    // Backup
    mctsBackup(node, reward);
  }
  
  // return best move from root (most visits or highest value)
  return root;
}

// =================== Monte-Carlo Playouts (lightweight) ===================
// random playout until termination or depth limit; returns winner (1/2/0 draw)
function randomPlayout(bd, toPlay){
  let sim = cloneBoard(bd);
  let player = toPlay;
  let steps = 0;
  while(true){
    // check terminal (win)
    // find any immediate winner quickly (small optimization)
    const winA = findImmediateWinningColsFor(sim,1);
    if(winA.length) return 1;
    const winB = findImmediateWinningColsFor(sim,2);
    if(winB.length) return 2;
    const valid = getValidMoves(sim);
    if(valid.length===0) return 0;
    // prevent extremely long playouts
    if(steps++ > MCTS_PLY_DEPTH) return 0;
    // simple rollout policy: prefer center-ish empty columns
    const center = Math.floor(COLS/2);
    // assemble weighted list
    const weights = valid.map(c => 1 + (COLS - Math.abs(center - c)));
    const sum = weights.reduce((a,b)=>a+b,0);
    let r = Math.random()*sum;
    let idx = 0;
    while(r > weights[idx]){ r -= weights[idx]; idx++; }
    const chosen = valid[idx];
    // apply
    for(let rr=ROWS-1; rr>=0; rr--) if(sim[rr][chosen]===0){ sim[rr][chosen] = player; break; }
    // check win
    // quick local check: the piece just placed could win; but we don't track the row here for speed — skip expensive checks
    player = player===1?2:1;
  }
}

// =================== Ultra decision maker (Pure MCTS) ===================
async function ultraChooseAndPlay(){
  if(gameOver) return;
  const aiPlayer = Number(aiPlayerSelect.value);
  const rating = Number(aiRatingInput.value || 1500);
  
  // 1) immediate win
  const immediate = findImmediateWinningColsFor(board, aiPlayer);
  if(immediate.length){ 
    handleColumnClick(immediate[0], true); 
    return; 
  }

  // 2) block opponent immediate win
  const opp = aiPlayer===1?2:1;
  const oppWin = findImmediateWinningColsFor(board, opp);
  if(oppWin.length){ 
    handleColumnClick(oppWin[0], true); 
    return; 
  }

  // 3) get valid moves
  const valid = getValidMoves(board);
  if(valid.length===0) return;

  // 3.5) if the live evaluator already has a credible top line, play its top move
  const persistedAnalysis = convertStoredAnalysisToPlayerPerspective(getKnownAnalysisSync(board, aiPlayer), aiPlayer);
  const activeAnalysis =
    latestAnalysis?.jobId === analysisJobId && latestAnalysis.positionKey === getAnalysisPositionKey(board, aiPlayer)
      ? latestAnalysis
      : persistedAnalysis
        ? {
            raw: aiPlayer === 1 ? persistedAnalysis.score : -persistedAnalysis.score,
            depth: persistedAnalysis.depth,
            pv: persistedAnalysis.pv.slice()
          }
        : null;
  const liveTopMove = activeAnalysis?.pv?.[0] ?? null;
  const liveLineIsUsable =
    Number.isInteger(liveTopMove) &&
    valid.includes(liveTopMove) &&
    (isForcedScore(activeAnalysis.raw) || activeAnalysis.depth >= 4);
  if (liveLineIsUsable) {
    handleColumnClick(liveTopMove, true);
    return;
  }

  // 4) run MCTS with rating-scaled simulations
  const numSimulations = Math.min(
    SIMULATION_LIMIT,
    Math.max(50, ratingToSimulations(rating))
  );

  try {
    const root = runMCTS(cloneBoard(board), aiPlayer, numSimulations, rating);

    // Ensure we have at least created some children
    if (!root || !root.children || root.children.length === 0) {
      // MCTS didn't create children, make random move
      const chosen = valid[Math.floor(Math.random() * valid.length)];
      handleColumnClick(chosen, true);
      return;
    }

    // 5) select best move by visits (or value if tied)
    // higher visits = more confident in this move's quality
    let bestChild = null;
    let bestScore = -Infinity;
    
    for(const child of root.children){
      // score = (value/visits) for exploitation, weighted by visits for confidence
      const avgValue = child.value / (child.visits || 1);
      const confidence = child.visits;
      const score = avgValue + 0.1 * Math.log(confidence + 1);
      
      if(score > bestScore){
        bestScore = score;
        bestChild = child;
      }
    }

    // handle tie-breaking: if multiple moves are very close, explore options
    if(root.children.length > 1){
      const sortedByVisits = [...root.children].sort((a,b)=> b.visits - a.visits);
      if(sortedByVisits.length >= 2){
        const topVisits = sortedByVisits[0].visits;
        const secondVisits = sortedByVisits[1].visits;
        
        // if top 2 are very close in visit count, consider randomizing for variety
        if(topVisits > 0 && Math.abs(topVisits - secondVisits) / topVisits < 0.1){
          if(RANDOM_MOVE_TIEBREAK && Math.random() < 0.15){
            bestChild = sortedByVisits[Math.random() < 0.5 ? 0 : 1];
          }
        }
      }
    }

    if(bestChild && bestChild.move !== undefined){
      handleColumnClick(bestChild.move, true);
    } else if(valid.length > 0) {
      // fallback: pick random valid move
      const chosen = valid[Math.floor(Math.random()*valid.length)];
      handleColumnClick(chosen, true);
    }
  } catch (e) {
    // safety fallback: make random move if MCTS errors
    console.error("MCTS error:", e);
    if(valid.length > 0) {
      const chosen = valid[Math.floor(Math.random()*valid.length)];
      handleColumnClick(chosen, true);
    }
  }
}

// =================== Lightweight helpers for UI & mapping rating ===================
function ratingToClass(r){
  r = Number(r);
  if(r < 300) return 'Class F';
  if(r < 600) return 'Class E';
  if(r < 900) return 'Class D';
  if(r < 1200) return 'Class C';
  if(r < 1500) return 'Class B';
  if(r < 1800) return 'Class A';
  if(r < 2100) return 'Class M';
  if(r < 2400) return 'Class GM';
  if(r < 2700) return 'Class SGM';
  return 'Class HGM';
}

function ratingToSimulations(r){
  // map rating to MCTS simulation budget
  r = Number(r) || 1500;
  if (r < 300) return 50;
  if (r < 600) return 100;
  if (r < 900) return 200;
  if (r < 1200) return 350;
  if (r < 1500) return 500;
  if (r < 1800) return 700;
  if (r < 2100) return 900;
  return 1200; // top difficulty
}

// =================== Initialization & events ===================
void openAnalysisDb();
createEmptyBoard();
renderBoard();
updateEvalAndThreats();

restartBtn?.addEventListener('click', ()=>{
  if(pendingAITimeout) {
    clearTimeout(pendingAITimeout);
    pendingAITimeout = null;
  }
  createEmptyBoard();
  currentPlayer = 1;
  gameOver = false;
  lastWinningCells = null;
  moveHistory = [];
  renderBoard();
  updateMessage();
  if(reviewEl){ reviewEl.hidden = true; reviewEl.innerHTML=''; }
  updateEvalAndThreats();
  triggerAITurnIfNeeded();
});

aiRatingInput?.addEventListener('input', ()=>{
  aiClassLabel.textContent = ratingToClass(aiRatingInput.value);
  updateEvalAndThreats();
});

aiToggle?.addEventListener('change', ()=>{
  updateEvalAndThreats();
  triggerAITurnIfNeeded();
});

aiPlayerSelect?.addEventListener('change', ()=>{
  updateEvalAndThreats();
  triggerAITurnIfNeeded();
});

document.addEventListener('keydown', (e)=>{
  if(e.key >= '1' && e.key <= '7'){
    const col = Number(e.key) - 1;
    handleColumnClick(col);
  }
});

// if AI selected for starting player on load
triggerAITurnIfNeeded();

// Expose a tiny API for console debugging (optional)
window.__cn4 = {
  board, createEmptyBoard, renderBoard, evaluateBoard, evaluateBoardForCyan, findImmediateWinningColsFor
};
