// app.js — UI & пайплайн
import { parseCSVFile, parseCSVUrl, inferSchema, buildVocab, prepareTensors } from './data_utils.js';
import { buildMLP, trainModel, evaluateModel } from './model.js';

const ui = {
  btnLoadEmbedded: document.getElementById('btnLoadEmbedded'),
  csv: document.getElementById('csv'),
  colTarget: document.getElementById('colTarget'),
  task: document.getElementById('task'),
  exclude: document.getElementById('exclude'),
  colText: document.getElementById('colText'),
  vocab: document.getElementById('vocab'),
  split: document.getElementById('split'),
  classWeight: document.getElementById('classWeight'),
  btnPrep: document.getElementById('btnPrep'),

  arch: document.getElementById('arch'),
  drop: document.getElementById('drop'),
  epochs: document.getElementById('epochs'),
  batch: document.getElementById('batch'),
  lr: document.getElementById('lr'),
  btnBuild: document.getElementById('btnBuild'),
  btnTrain: document.getElementById('btnTrain'),
  btnEval: document.getElementById('btnEval'),

  meta: document.getElementById('meta'),
  results: document.getElementById('results'),
  thead: document.getElementById('thead'),
  tbody: document.getElementById('tbody'),
  trainChart: document.getElementById('trainChart'),
  log: document.getElementById('log'),
};

let RAW=null, SCHEMA=null, DATA=null, MODEL=null, CHART=null;
let TEXT_VOCAB=null, TEXT_COL=null;

const logln = s => { ui.log.textContent += s + "\n"; ui.log.scrollTop = ui.log.scrollHeight; };
function disable(b){
  ui.btnPrep.disabled = b || !RAW;
  ui.btnBuild.disabled= b || !DATA;
  ui.btnTrain.disabled= b || !MODEL || !DATA;
  ui.btnEval.disabled = b || !MODEL || !DATA;
}

ui.btnLoadEmbedded.onclick = async ()=>{
  try{
    ui.log.textContent="";
    RAW = await parseCSVUrl('./data/wine_food_pairings.csv');
    afterLoad();
  }catch(e){ logln("❌ Ошибка загрузки: " + (e.message||e)); }
};

ui.csv.onchange = async ()=>{
  const f = ui.csv.files?.[0]; if(!f) return;
  ui.log.textContent = "";
  RAW = await parseCSVFile(f);
  afterLoad();
};

function afterLoad(){
  logln(`Строк: ${RAW.length}`);
  SCHEMA = inferSchema(RAW);
  const cols = Object.keys(SCHEMA);

  ui.colTarget.innerHTML = cols.map(c=> `<option value="${c}">${c}</option>`).join('');
  ui.exclude.innerHTML   = cols.map(c=> `<option value="${c}">${c}</option>`).join('');
  ui.colText.innerHTML   = ['<option value="">(нет)</option>']
    .concat(cols.filter(c=> SCHEMA[c].kind==='text').map(c=> `<option value="${c}">${c}</option>`))
    .join('');

  // догадаться о целевой (например, 'pairing'/'quality'/'score')
  const guess = cols.find(c=> /pair|score|rating|quality/i.test(c)) || cols[0];
  ui.colTarget.value = guess;

  // мета
  const numCnt = cols.filter(c=> SCHEMA[c].kind==='numeric').length;
  const catCnt = cols.filter(c=> SCHEMA[c].kind==='categorical').length;
  const txtCnt = cols.filter(c=> SCHEMA[c].kind==='text').length;
  ui.meta.innerHTML = `<span class="pill">Колонки: ${cols.length}</span> <span class="pill">Num: ${numCnt}</span> <span class="pill">Cat: ${catCnt}</span> <span class="pill">Text: ${txtCnt}</span>`;

  ui.btnPrep.disabled = false;
}

ui.btnPrep.onclick = async ()=>{
  try{
    disable(true);
    await initTF();

    const targetCol = ui.colTarget.value;
    const allCols = Object.keys(SCHEMA);
    const excluded = [...ui.exclude.selectedOptions].map(o=> o.value);

    TEXT_COL = ui.colText.value || null;
    TEXT_VOCAB = null;
    if (TEXT_COL){
      const V = Math.max(100, Math.min(5000, +ui.vocab.value||500));
      logln(`Строю словарь по колонке "${TEXT_COL}" (V=${V})…`);
      TEXT_VOCAB = buildVocab(RAW, TEXT_COL, V);
      logln(`Словарь готов: ${TEXT_VOCAB.size} токенов.`);
    }

    // признаки = все, кроме цели и исключённых (текстовую тоже включаем — она будет преобразована в BoW)
    const featureCols = allCols.filter(c=> c!==targetCol && !excluded.includes(c));

    const splitPct = Math.max(0.5, Math.min(0.95, (+ui.split.value||80)/100));
    const task = ui.task.value;
    const cWeight = ui.classWeight.value;

    DATA = prepareTensors(
      RAW, featureCols, targetCol, task, SCHEMA, splitPct, cWeight,
      TEXT_COL, TEXT_VOCAB
    );

    ui.meta.innerHTML += ` <span class="pill">Train: ${DATA.Xtrain.shape[0]}</span> <span class="pill">Test: ${DATA.Xtest.shape[0]}</span> <span class="pill">Inputs: ${DATA.inputDim}</span>`;
    logln(`Готово. Задача: ${DATA.isClassification? 'классификация':'регрессия'}.`);
    ui.btnBuild.disabled = false;
  } catch(e){
    logln("❌ Ошибка подготовки: " + (e.message||e));
  } finally { disable(false); }
};

ui.btnBuild.onclick = ()=>{
  try{
    disable(true);
    if (MODEL) MODEL.dispose();
    const cfg = {
      arch: ui.arch.value,
      drop: +ui.drop.value || 0,
      task: DATA.isClassification ? 'clf' : 'reg',
      nClasses: DATA.nClasses || 1,
      lr: +ui.lr.value || 1e-3
    };
    MODEL = buildMLP(DATA.inputDim, cfg);
    logln(`Модель: MLP ${cfg.arch}, drop=${cfg.drop}, task=${cfg.task}${cfg.task==='clf'? `, classes=${cfg.nClasses}`:''}`);
    ui.btnTrain.disabled = false;
  } finally { disable(false); }
};

ui.btnTrain.onclick = async ()=>{
  try{
    disable(true);
    initChart();
    const ep = +ui.epochs.value||20, bs= +ui.batch.value||64;
    logln(`Обучение: epochs=${ep}, batch=${bs}`);
    await trainModel(MODEL, DATA.Xtrain, DATA.Ytrain, ep, bs,
      (e,logs)=> addPoint(e+1, logs.loss, logs.val_loss, logs.acc??logs.accuracy??logs.mae, logs.val_acc??logs.val_accuracy??logs.val_mae));
    logln("Обучение завершено.");
    ui.btnEval.disabled = false;
  } catch(e){
    logln("❌ Ошибка обучения: " + (e.message||e));
  } finally { disable(false); }
};

ui.btnEval.onclick = async ()=>{
  try{
    disable(true);
    const res = await evaluateModel(MODEL, DATA.Xtest, DATA.Ytest, DATA.isClassification? 'clf':'reg');
    if (DATA.isClassification){
      ui.results.textContent = `Test — loss=${res.loss.toFixed(4)}, accuracy=${(res.acc*100).toFixed(1)}%`;
      await renderTopK(12);
    } else {
      ui.results.textContent = `Test — MAE=${res.mae.toFixed(3)} (loss=${res.loss.toFixed(3)})`;
      await renderPredVsTrue(20);
    }
  } catch(e){
    logln("❌ Ошибка оценки: " + (e.message||e));
  } finally { disable(false); }
};

/* helpers */
async function initTF(){
  try {
    tf.env().set('WEBGL_VERSION', 1);
    tf.env().set('WEBGL_PACK', false);
    await tf.setBackend('webgl'); await tf.ready();
  } catch { await tf.setBackend('cpu'); await tf.ready(); }
  logln("TF backend: " + tf.getBackend());
}
let CHART;
function initChart(){
  if (CHART) CHART.destroy();
  CHART = new Chart(ui.trainChart.getContext('2d'), {
    type:'line',
    data:{labels:[], datasets:[
      {label:'loss', data:[], tension:.2},
      {label:'val_loss', data:[], tension:.2},
      {label:'acc/mae', data:[], tension:.2},
      {label:'val_acc/val_mae', data:[], tension:.2},
    ]},
    options:{responsive:true, scales:{y:{beginAtZero:true}}, plugins:{legend:{position:'bottom'}}}
  });
}
function addPoint(epoch, loss, vloss, acc, vacc){
  CHART.data.labels.push(String(epoch));
  CHART.data.datasets[0].data.push(loss ?? null);
  CHART.data.datasets[1].data.push(vloss ?? null);
  CHART.data.datasets[2].data.push(acc ?? null);
  CHART.data.datasets[3].data.push(vacc ?? null);
  CHART.update();
}

async function renderTopK(K=12){
  ui.thead.innerHTML = `<tr><th>#</th><th>True</th><th>Pred</th><th>Prob</th></tr>`;
  const preds = MODEL.predict(DATA.Xtest);
  const P = await preds.array(); preds.dispose();
  const Y = await DATA.Ytest.array();
  const inv = Object.entries(DATA.labelMap).reduce((acc,[k,v])=> (acc[v]=k, acc), {});
  const rows=[];
  for (let i=0;i<Math.min(K,P.length);i++){
    const pi = P[i], yi = Y[i];
    const predIdx = pi.indexOf(Math.max(...pi));
    const trueIdx = yi.indexOf(1);
    rows.push(`<tr><td>${i+1}</td><td>${inv[trueIdx]}</td><td>${inv[predIdx]}</td><td>${(pi[predIdx]*100).toFixed(1)}%</td></tr>`);
  }
  ui.tbody.innerHTML = rows.join("");
}
async function renderPredVsTrue(K=20){
  ui.thead.innerHTML = `<tr><th>#</th><th>True</th><th>Pred</th></tr>`;
  const preds = MODEL.predict(DATA.Xtest);
  const P = await preds.array(); preds.dispose();
  const Y = await DATA.Ytest.array();
  const rows=[];
  for (let i=0;i<Math.min(K,P.length);i++){
    rows.push(`<tr><td>${i+1}</td><td>${Y[i][0].toFixed(3)}</td><td>${P[i][0].toFixed(3)}</td></tr>`);
  }
  ui.tbody.innerHTML = rows.join("");
}
