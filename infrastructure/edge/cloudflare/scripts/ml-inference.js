// ML Inference at Edge - WebAssembly-based model inference
import * as tf from '@tensorflow/tfjs';

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const startTime = Date.now();
    
    try {
      // Parse request
      const requestData = await parseRequest(request);
      
      // Select model based on endpoint
      const modelName = getModelName(url.pathname);
      
      // Load or get cached model
      const model = await loadModel(modelName, env);
      
      // Preprocess input
      const preprocessed = await preprocessInput(requestData, modelName);
      
      // Run inference
      const prediction = await runInference(model, preprocessed, env);
      
      // Postprocess output
      const result = await postprocessOutput(prediction, modelName);
      
      // Cache result if applicable
      if (shouldCache(modelName)) {
        const cacheKey = generateCacheKey(requestData, modelName);
        ctx.waitUntil(
          env.MODEL_CACHE.put(cacheKey, JSON.stringify(result), {
            expirationTtl: 300 // 5 minutes
          })
        );
      }
      
      // Log metrics
      const duration = Date.now() - startTime;
      ctx.waitUntil(logInferenceMetrics(modelName, duration, env));
      
      return new Response(JSON.stringify({
        model: modelName,
        prediction: result,
        confidence: result.confidence || null,
        inference_time_ms: duration,
        edge_location: request.cf?.colo || 'UNKNOWN'
      }), {
        headers: {
          'Content-Type': 'application/json',
          'X-Model-Version': model.version,
          'X-Inference-Time': `${duration}ms`
        }
      });
      
    } catch (error) {
      console.error('Inference error:', error);
      
      // Fallback to remote inference
      if (env.FALLBACK_INFERENCE_URL) {
        return await fallbackInference(request, env);
      }
      
      return new Response(JSON.stringify({
        error: 'Inference failed',
        message: error.message
      }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  }
};

async function parseRequest(request) {
  const contentType = request.headers.get('Content-Type');
  
  if (contentType?.includes('application/json')) {
    return await request.json();
  } else if (contentType?.includes('multipart/form-data')) {
    const formData = await request.formData();
    const data = {};
    
    for (const [key, value] of formData.entries()) {
      if (value instanceof File) {
        data[key] = await value.arrayBuffer();
      } else {
        data[key] = value;
      }
    }
    
    return data;
  } else if (contentType?.includes('image/')) {
    return {
      image: await request.arrayBuffer(),
      type: contentType
    };
  }
  
  return await request.text();
}

function getModelName(pathname) {
  const segments = pathname.split('/');
  const modelMap = {
    'classify': 'image-classifier-v2',
    'detect': 'object-detector-v1',
    'sentiment': 'sentiment-analyzer-v3',
    'translate': 'translator-v1',
    'summarize': 'text-summarizer-v2',
    'recommend': 'recommendation-engine-v4',
    'anomaly': 'anomaly-detector-v1',
    'forecast': 'time-series-forecaster-v2'
  };
  
  const endpoint = segments[segments.length - 1];
  return modelMap[endpoint] || 'default-model';
}

async function loadModel(modelName, env) {
  // Check if model is already loaded in memory
  if (global.models && global.models[modelName]) {
    return global.models[modelName];
  }
  
  // Check cache for model weights
  const cacheKey = `model:${modelName}`;
  const cachedModel = await env.MODEL_CACHE.get(cacheKey, 'arrayBuffer');
  
  if (cachedModel) {
    // Load from cache
    const model = await loadModelFromBuffer(cachedModel, modelName);
    
    // Store in memory
    if (!global.models) global.models = {};
    global.models[modelName] = model;
    
    return model;
  }
  
  // Load from WebAssembly module or remote
  const model = await loadModelFromSource(modelName, env);
  
  // Cache model weights
  const modelBuffer = await serializeModel(model);
  await env.MODEL_CACHE.put(cacheKey, modelBuffer, {
    expirationTtl: 86400 // 24 hours
  });
  
  // Store in memory
  if (!global.models) global.models = {};
  global.models[modelName] = model;
  
  return model;
}

async function loadModelFromBuffer(buffer, modelName) {
  // For WebAssembly models
  if (modelName.includes('wasm')) {
    const wasmModule = await WebAssembly.instantiate(buffer);
    return {
      type: 'wasm',
      module: wasmModule.instance,
      version: '1.0.0',
      predict: wasmModule.instance.exports.predict
    };
  }
  
  // For TensorFlow.js models
  const modelData = new Uint8Array(buffer);
  const model = await tf.loadLayersModel(tf.io.fromMemory(modelData));
  
  return {
    type: 'tfjs',
    model: model,
    version: '1.0.0',
    predict: async (input) => model.predict(input)
  };
}

async function loadModelFromSource(modelName, env) {
  // Load quantized model for edge deployment
  const modelUrl = `${env.MODEL_REGISTRY_URL}/${modelName}/quantized`;
  
  try {
    const response = await fetch(modelUrl);
    if (!response.ok) throw new Error('Model not found');
    
    const modelBuffer = await response.arrayBuffer();
    return await loadModelFromBuffer(modelBuffer, modelName);
    
  } catch (error) {
    // Load default lightweight model
    return getDefaultModel(modelName);
  }
}

function getDefaultModel(modelName) {
  // Return a simple rule-based model as fallback
  return {
    type: 'rule-based',
    version: 'fallback',
    predict: async (input) => {
      // Simple heuristic-based predictions
      if (modelName.includes('sentiment')) {
        return sentimentHeuristic(input);
      } else if (modelName.includes('classify')) {
        return classificationHeuristic(input);
      }
      
      return { result: 'default', confidence: 0.5 };
    }
  };
}

async function preprocessInput(data, modelName) {
  if (modelName.includes('image')) {
    return await preprocessImage(data);
  } else if (modelName.includes('text') || modelName.includes('sentiment')) {
    return await preprocessText(data);
  } else if (modelName.includes('time-series')) {
    return await preprocessTimeSeries(data);
  }
  
  return data;
}

async function preprocessImage(data) {
  let imageBuffer;
  
  if (data.image) {
    imageBuffer = data.image;
  } else if (data.url) {
    const response = await fetch(data.url);
    imageBuffer = await response.arrayBuffer();
  } else if (data.base64) {
    imageBuffer = base64ToArrayBuffer(data.base64);
  } else {
    throw new Error('No image data provided');
  }
  
  // Convert to tensor and resize
  const imageTensor = tf.node.decodeImage(new Uint8Array(imageBuffer));
  const resized = tf.image.resizeBilinear(imageTensor, [224, 224]);
  const normalized = resized.div(255.0);
  const batched = normalized.expandDims(.0);
  
  return batched;
}

async function preprocessText(data) {
  const text = data.text || data;
  
  // Tokenization
  const tokens = tokenize(text);
  
  // Convert to embeddings (simplified)
  const embeddings = tokens.map(token => getTokenEmbedding(token));
  
  // Pad or truncate to fixed length
  const maxLength = 512;
  const padded = padSequence(embeddings, maxLength);
  
  return tf.tensor2d([padded]);
}

async function preprocessTimeSeries(data) {
  const series = data.values || data;
  
  // Normalize
  const mean = series.reduce((a, b) => a + b) / series.length;
  const std = Math.sqrt(
    series.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / series.length
  );
  
  const normalized = series.map(val => (val - mean) / std);
  
  // Create sliding windows
  const windowSize = data.window_size || 24;
  const windows = [];
  
  for (let i = 0; i <= normalized.length - windowSize; i++) {
    windows.push(normalized.slice(i, i + windowSize));
  }
  
  return tf.tensor3d([windows]);
}

async function runInference(model, input, env) {
  try {
    // Check if we should use federated learning update
    if (model.supportsFederated) {
      await updateFederatedModel(model, env);
    }
    
    // Run prediction
    const prediction = await model.predict(input);
    
    // Convert tensor to array if needed
    if (prediction instanceof tf.Tensor) {
      return await prediction.array();
    }
    
    return prediction;
    
  } catch (error) {
    console.error('Inference execution error:', error);
    throw error;
  }
}

async function postprocessOutput(prediction, modelName) {
  if (modelName.includes('classify')) {
    return postprocessClassification(prediction);
  } else if (modelName.includes('detect')) {
    return postprocessDetection(prediction);
  } else if (modelName.includes('sentiment')) {
    return postprocessSentiment(prediction);
  } else if (modelName.includes('recommend')) {
    return postprocessRecommendations(prediction);
  }
  
  return { result: prediction };
}

function postprocessClassification(prediction) {
  const classes = [
    'cat', 'dog', 'bird', 'car', 'person',
    'bicycle', 'airplane', 'ship', 'truck', 'horse'
  ];
  
  const probabilities = Array.isArray(prediction[0]) ? prediction[0] : prediction;
  const maxIndex = probabilities.indexOf(Math.max(...probabilities));
  
  return {
    class: classes[maxIndex] || 'unknown',
    confidence: probabilities[maxIndex],
    probabilities: classes.map((cls, idx) => ({
      class: cls,
      probability: probabilities[idx]
    })).sort((a, b) => b.probability - a.probability).slice(0, 5)
  };
}

function postprocessDetection(prediction) {
  // Parse bounding boxes and classes
  const [boxes, scores, classes] = prediction;
  const detections = [];
  
  for (let i = 0; i < scores.length; i++) {
    if (scores[i] > 0.5) {
      detections.push({
        bbox: boxes[i],
        score: scores[i],
        class: getClassName(classes[i])
      });
    }
  }
  
  return {
    detections: detections,
    count: detections.length
  };
}

function postprocessSentiment(prediction) {
  const sentiments = ['negative', 'neutral', 'positive'];
  const scores = Array.isArray(prediction[0]) ? prediction[0] : prediction;
  const maxIndex = scores.indexOf(Math.max(...scores));
  
  return {
    sentiment: sentiments[maxIndex],
    confidence: scores[maxIndex],
    scores: {
      negative: scores[0],
      neutral: scores[1],
      positive: scores[2]
    }
  };
}

function postprocessRecommendations(prediction) {
  const items = Array.isArray(prediction[0]) ? prediction[0] : prediction;
  
  return {
    recommendations: items
      .map((score, idx) => ({ id: idx, score: score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 10)
      .map(item => ({
        item_id: `item_${item.id}`,
        score: item.score,
        confidence: item.score
      }))
  };
}

function shouldCache(modelName) {
  // Cache results for deterministic models
  const cacheableModels = [
    'image-classifier',
    'object-detector',
    'sentiment-analyzer',
    'text-summarizer'
  ];
  
  return cacheableModels.some(model => modelName.includes(model));
}

function generateCacheKey(data, modelName) {
  const hash = crypto.createHash('sha256');
  hash.update(JSON.stringify(data));
  hash.update(modelName);
  return `inference:${hash.digest('hex')}`;
}

async function fallbackInference(request, env) {
  try {
    const fallbackUrl = new URL(env.FALLBACK_INFERENCE_URL);
    fallbackUrl.pathname = new URL(request.url).pathname;
    
    const fallbackRequest = new Request(fallbackUrl, request);
    const response = await fetch(fallbackRequest);
    
    // Add header to indicate fallback was used
    const modifiedResponse = new Response(response.body, response);
    modifiedResponse.headers.set('X-Inference-Mode', 'fallback');
    
    return modifiedResponse;
    
  } catch (error) {
    return new Response(JSON.stringify({
      error: 'Both edge and fallback inference failed',
      message: error.message
    }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

async function updateFederatedModel(model, env) {
  // Check if update is needed
  const lastUpdate = await env.MODEL_CACHE.get(`federated:${model.name}:last_update`);
  const now = Date.now();
  
  if (lastUpdate && (now - parseInt(lastUpdate)) < 3600000) {
    return; // Updated less than an hour ago
  }
  
  try {
    // Fetch aggregated updates from federated learning server
    const response = await fetch(`${env.FL_SERVER_URL}/models/${model.name}/updates`);
    if (!response.ok) return;
    
    const updates = await response.json();
    
    // Apply updates to model weights
    if (updates.weights) {
      await applyWeightUpdates(model, updates.weights);
    }
    
    // Update timestamp
    await env.MODEL_CACHE.put(
      `federated:${model.name}:last_update`,
      now.toString(),
      { expirationTtl: 86400 }
    );
    
  } catch (error) {
    console.error('Federated update failed:', error);
  }
}

async function applyWeightUpdates(model, weights) {
  if (model.type === 'tfjs') {
    // Update TensorFlow.js model weights
    const weightTensors = weights.map(w => tf.tensor(w.values, w.shape));
    model.model.setWeights(weightTensors);
  } else if (model.type === 'wasm') {
    // Update WebAssembly model memory
    const memory = model.module.exports.memory;
    const weightBuffer = new Float32Array(memory.buffer);
    
    let offset = 0;
    for (const weight of weights) {
      weightBuffer.set(weight.values, offset);
      offset += weight.values.length;
    }
  }
}

async function serializeModel(model) {
  if (model.type === 'tfjs') {
    // Serialize TensorFlow.js model
    const saveResult = await model.model.save(tf.io.withSaveHandler(async (artifacts) => {
      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: 'JSON'
        }
      };
    }));
    
    return new TextEncoder().encode(JSON.stringify(saveResult));
  } else if (model.type === 'wasm') {
    // Return WebAssembly module buffer
    return model.module.exports.memory.buffer;
  }
  
  return new ArrayBuffer(0);
}

async function logInferenceMetrics(modelName, duration, env) {
  const metrics = {
    timestamp: Date.now(),
    model: modelName,
    duration: duration,
    success: true
  };
  
  // Store metrics for aggregation
  const key = `inference:metrics:${Date.now()}`;
  await env.MODEL_CACHE.put(key, JSON.stringify(metrics), {
    expirationTtl: 3600
  });
}

// Utility functions
function tokenize(text) {
  // Simple tokenization
  return text.toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter(token => token.length > 0);
}

function getTokenEmbedding(token) {
  // Simple hash-based embedding (replace with real embeddings)
  let hash = 0;
  for (let i = 0; i < token.length; i++) {
    hash = ((hash << 5) - hash) + token.charCodeAt(i);
    hash = hash & hash;
  }
  
  // Generate pseudo-embedding vector
  const embedding = new Array(128);
  for (let i = 0; i < 128; i++) {
    embedding[i] = ((hash * (i + 1)) % 200 - 100) / 100;
  }
  
  return embedding;
}

function padSequence(sequence, maxLength) {
  if (sequence.length > maxLength) {
    return sequence.slice(0, maxLength);
  }
  
  const padded = [...sequence];
  const padding = new Array(128).fill(0);
  
  while (padded.length < maxLength) {
    padded.push(padding);
  }
  
  return padded.flat();
}

function base64ToArrayBuffer(base64) {
  const binary = atob(base64);
  const buffer = new ArrayBuffer(binary.length);
  const bytes = new Uint8Array(buffer);
  
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  
  return buffer;
}

function getClassName(classId) {
  const classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light'
  ];
  
  return classes[classId] || `class_${classId}`;
}

function sentimentHeuristic(input) {
  const text = input.text || input;
  const positiveWords = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love'];
  const negativeWords = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'poor'];
  
  const words = text.toLowerCase().split(/\s+/);
  let positiveCount = 0;
  let negativeCount = 0;
  
  words.forEach(word => {
    if (positiveWords.includes(word)) positiveCount++;
    if (negativeWords.includes(word)) negativeCount++;
  });
  
  if (positiveCount > negativeCount) {
    return { sentiment: 'positive', confidence: 0.7 };
  } else if (negativeCount > positiveCount) {
    return { sentiment: 'negative', confidence: 0.7 };
  }
  
  return { sentiment: 'neutral', confidence: 0.6 };
}

function classificationHeuristic(input) {
  // Simple rule-based classification
  return {
    class: 'unknown',
    confidence: 0.3,
    probabilities: [
      { class: 'unknown', probability: 0.3 },
      { class: 'other', probability: 0.7 }
    ]
  };
}