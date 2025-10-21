// model.js — простой MLP для табличных + текст-BoW

export function buildMLP(inputDim, {arch='128-64', drop=0.2, task='clf', nClasses=2, lr=1e-3}){
  const m = tf.sequential();
  const sizes = arch.split('-').map(s=> +s);

  m.add(tf.layers.dense({inputShape:[inputDim], units:sizes[0], activation:'relu'}));
  if (drop>0) m.add(tf.layers.dropout({rate:drop}));
  for (let i=1;i<sizes.length;i++){
    m.add(tf.layers.dense({units:sizes[i], activation:'relu'}));
    if (drop>0) m.add(tf.layers.dropout({rate:drop}));
  }

  if (task==='clf'){
    m.add(tf.layers.dense({units:nClasses, activation:'softmax'}));
    m.compile({optimizer: tf.train.adam(lr), loss: 'categoricalCrossentropy', metrics:['accuracy']});
  } else {
    m.add(tf.layers.dense({units:1, activation:'linear'}));
    m.compile({optimizer: tf.train.adam(lr), loss: 'meanAbsoluteError', metrics:['mae']});
  }
  return m;
}

export async function trainModel(model, Xtrain, Ytrain, epochs=20, batch=64, onEpoch){
  return model.fit(Xtrain, Ytrain, {
    epochs, batchSize: batch, validationSplit: 0.1, shuffle:true,
    callbacks: { onEpochEnd: async (ep,logs)=> onEpoch?.(ep,logs) }
  });
}

export async function evaluateModel(model, Xtest, Ytest, task){
  const ev = await model.evaluate(Xtest, Ytest, {verbose:0});
  if (task==='clf'){
    const [lossT, accT] = await Promise.all(ev.map(t=> t.data()));
    return { loss: lossT[0], acc: accT[0] };
  } else {
    const [lossT, maeT] = await Promise.all(ev.map(t=> t.data()));
    return { loss: lossT[0], mae: maeT[0] };
  }
}
