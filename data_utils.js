// data_utils.js — CSV → фичи: numeric (min-max), categorical (one-hot), text (bag-of-words)

export async function parseCSVFile(file){
  return new Promise((resolve,reject)=>{
    Papa.parse(file, { header:true, dynamicTyping:true, skipEmptyLines:true,
      complete: res => resolve(res.data), error: reject });
  });
}

export async function parseCSVUrl(url){
  return new Promise((resolve,reject)=>{
    Papa.parse(url, { download:true, header:true, dynamicTyping:true, skipEmptyLines:true,
      complete: res => resolve(res.data), error: reject });
  });
}

export function inferSchema(rows){
  const cols = Object.keys(rows[0] || {});
  const schema = {};
  for (const c of cols){
    const vals = rows.slice(0, Math.min(500, rows.length)).map(r=> r[c]);
    const nums = vals.filter(v=> typeof v === 'number' && Number.isFinite(v));
    const uniq = new Set(vals.filter(v=> v!==null && v!==undefined).map(v=> String(v)));
    // если слишком длинные строки — считаем текстом
    const longTextShare = vals.filter(v=> (typeof v === 'string') && v.length > 20).length / (vals.length||1);
    const kind = longTextShare > 0.3 ? 'text' : ((nums.length/vals.length > 0.7) ? 'numeric' : 'categorical');
    schema[c] = { kind, unique: uniq.size };
  }
  return schema;
}

// простой токенайзер
function tokenize(s){
  return String(s || '')
    .toLowerCase()
    .replace(/[^a-zа-яё0-9\s]/gi, ' ')
    .split(/\s+/)
    .filter(t=> t && t.length>1);
}

// строим словарь для текстовой колонки
export function buildVocab(rows, textCol, vocabSize=500){
  const freq = new Map();
  for (const r of rows){
    for (const t of tokenize(r[textCol])) freq.set(t, (freq.get(t)||0)+1);
  }
  // отсортируем и возьмём top-K
  const vocab = [...freq.entries()].sort((a,b)=> b[1]-a[1]).slice(0, vocabSize).map(([w])=> w);
  const index = new Map(vocab.map((w,i)=> [w,i]));
  return { vocab, index, size: vocab.length };
}

// текст → вектор (binary presence)
export function textToVec(row, textCol, index, size){
  const v = new Float32Array(size);
  const seen = new Set();
  for (const t of tokenize(row[textCol])){
    const i = index.get(t);
    if (i!==undefined && !seen.has(i)){ v[i]=1; seen.add(i); }
  }
  return v;
}

// препроцесс табличных (без текста)
export function buildTabularPreprocessor(rows, featureCols, schema){
  const catMaps = {}; // {col: [cats]}
  const numStats = {}; // {col: {min,max}}

  for (const c of featureCols){
    if (schema[c].kind === 'categorical'){
      const cats = [...new Set(rows.map(r=> r[c]).map(v=> String(v)))].filter(v=> v!=='undefined' && v!=='null');
      catMaps[c] = cats.slice(0, 200); // cap
    } else if (schema[c].kind === 'numeric') {
      const vals = rows.map(r=> +r[c]).filter(Number.isFinite);
      const min = Math.min(...vals), max = Math.max(...vals);
      numStats[c] = {min, max};
    }
  }

  let dim = 0;
  for (const c of featureCols){
    if (schema[c].kind === 'categorical') dim += catMaps[c].length;
    else if (schema[c].kind === 'numeric') dim += 1;
  }

  function transformRow(r){
    const out = new Float32Array(dim);
    let k = 0;
    for (const c of featureCols){
      if (schema[c].kind === 'categorical'){
        const cats = catMaps[c]; const v = String(r[c]);
        for (let i=0;i<cats.length;i++) out[k+i] = (cats[i]===v) ? 1 : 0;
        k += cats.length;
      } else if (schema[c].kind === 'numeric'){
        const {min,max} = numStats[c] || {min:0,max:1};
        const x = Number.isFinite(+r[c]) ? (+r[c] - min) / (max - min + 1e-9) : 0;
        out[k++] = x;
      }
    }
    return out;
  }

  return { dim, catMaps, numStats, transformRow };
}

// готовим X/y (включая текстовую колонку, если указана)
export function prepareTensors(rows, featureCols, targetCol, task, schema, splitPct=0.8, classWeight='off', textCol=null, vocabConf=null){
  const hasText = textCol && schema[textCol]?.kind === 'text' && vocabConf;
  const preTab = buildTabularPreprocessor(rows, featureCols.filter(c=> c!==textCol), schema);
  const preText = hasText ? vocabConf : null;

  // какое задание
  const autoClf = schema[targetCol].kind!=='numeric' && schema[targetCol].unique<=20;
  const isClassification = (task==='clf') || (task==='auto' && autoClf);

  // карта меток
  let labelMap=null, nClasses=0;
  if (isClassification){
    const cats = [...new Set(rows.map(r=> String(r[targetCol])))];
    labelMap = Object.fromEntries(cats.map((v,i)=> [v,i]));
    nClasses = cats.length;
  }

  // перемешаем
  const shuffled = rows.slice();
  for (let i=shuffled.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [shuffled[i],shuffled[j]]=[shuffled[j],shuffled[i]]; }

  // X/y
  const X=[], y=[];
  for (const r of shuffled){
    const tab = preTab.transformRow(r);             // табличные
    const txt = hasText ? textToVec(r, textCol, preText.index, preText.size) : new Float32Array(0);
    const xi = new Float32Array(preTab.dim + txt.length);
    xi.set(tab,0); if (txt.length) xi.set(txt, preTab.dim);
    X.push(xi);

    if (isClassification){
      const yi = new Float32Array(nClasses); yi[labelMap[String(r[targetCol])]] = 1;
      y.push(yi);
    } else {
      const val = Number(r[targetCol]);
      y.push([Number.isFinite(val) ? val : 0]);
    }
  }

  const N = X.length, split = Math.floor(N*splitPct);
  const inDim = preTab.dim + (preText?.size||0);
  const Xt = tf.tensor2d(X.slice(0,split), [split, inDim]);
  const Yt = tf.tensor2d(y.slice(0,split), [split, isClassification? nClasses: 1]);
  const Xv = tf.tensor2d(X.slice(split), [N-split, inDim]);
  const Yv = tf.tensor2d(y.slice(split), [N-split, isClassification? nClasses: 1]);

  // веса классов (простой авто-подсчёт)
  let classWeights=null;
  if (isClassification && classWeight==='auto'){
    const counts = new Array(nClasses).fill(0);
    for (let i=0;i<split;i++){ counts[y[i].findIndex(v=> v===1)]++; }
    const maxc = Math.max(...counts);
    classWeights = counts.map(c=> c>0 ? maxc/c : 1);
  }

  return {
    Xtrain:Xt, Ytrain:Yt, Xtest:Xv, Ytest:Yv,
    isClassification, nClasses, labelMap,
    inputDim: inDim
  };
}
