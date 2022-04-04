public class BackpropagationNN {
  
  public init(ephos: Int, learningRate: Float, minRange: Float = -1, maxRange: Float = 1) {
    self.ephos = ephos
    self.learningRate = learningRate
    self.minRange = minRange
    self.maxRange = maxRange
  }
  
  public let minRange: Float
  public let maxRange: Float
  public let ephos: Int
  public let learningRate: Float
  
  public var onEphos: ((Int) -> ())?
  public var onMSE: ((Float) -> ())?
  
  public var inputLayer: Layer = Layer()
  public var hiddenLayer: Layer = Layer()
  public var outputLayer: Layer = {
    let layer = Layer()
    layer.neurons.append(Neuron())
    return layer
  }()
  
  public func generateRange() -> Float {
    return .random(in: minRange...maxRange)
  }
  
  public func createLayers(inputNeurons: Int, hiddenNeurons: Int) {
    for _ in 0..<inputNeurons {
      let neuron = Neuron()
      for _ in 0..<hiddenNeurons {
        neuron.weights.append(generateRange())
      }
      inputLayer.neurons.append(neuron)
    }
    for _ in 0..<hiddenNeurons {
      let neuron = Neuron()
      neuron.weights.append(generateRange())
      hiddenLayer.neurons.append(neuron)
    }
  }
  
  public func predict(values: [Float]) -> [Float] {
    return forward(inputValues: values)
  }
  
  public func startTraining(data: [DataSet]) {
    for e in 0..<ephos {
      var predicts: [Float] = []
      var expectedResults: [Float] = []
      for d in data {
        train(values: d.input, expected: d.expected)
        guard let firstPredict = predict(values: d.input).first else { return }
        predicts.append(firstPredict)
        expectedResults.append(d.expected)
      }
      onEphos?(e)
      onMSE?(MSE(predicts: predicts, expectedResults: expectedResults))
    }
    print("🟢 Training is over")
  }
  
  public func train(values: [Float], expected: Float) {
    let forward = forward(inputValues: values)
    guard let actual = forward.first else { return }
    let error: Float = actual - expected
    let sigmoidDx = actual * (1 - actual)
    let weightsDelta = error * sigmoidDx
    hiddenLayer.backward(weightDelta: weightsDelta, learningRate: learningRate)
    for i in 0..<inputLayer.neurons.count {
      for w in 0..<inputLayer.neurons[i].weights.count {
        inputLayer.neurons[i].weights[w] = inputLayer.neurons[i].weights[w] - inputLayer.neurons[i].value * hiddenLayer.neurons[w].weightsDelta * learningRate
      }
    }
  }
  
  private func forward(inputValues: [Float]) -> [Float] {
    inputLayer.set(values: inputValues)
    let forwardResults = inputLayer.forward(nextLayerNeurons: hiddenLayer.neurons.count)
    let sigmoidValues = forwardResults.map { ActivationFunction.sigmoid(x: $0) }
    hiddenLayer.set(values: sigmoidValues)
    let finalResults = hiddenLayer.forward(nextLayerNeurons: outputLayer.neurons.count)
    let finalSigmoidResults = finalResults.map { ActivationFunction.sigmoid(x: $0) }
    outputLayer.set(values: finalSigmoidResults)
    return finalSigmoidResults
  }
}
