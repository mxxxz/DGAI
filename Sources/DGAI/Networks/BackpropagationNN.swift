public class BackpropagationNN {
  
  public init(ephos: Int, learningRate: Float, minRange: Float = -1, maxRange: Float = 1) {
    self.ephos = ephos
    self.learningRate = learningRate
    self.minRange = minRange
    self.maxRange = maxRange
  }
  
  private let minRange: Float
  private let maxRange: Float
  private let ephos: Int
  private let learningRate: Float
  
  public var inputLayer: Layer = Layer()
  public var hiddenLayer: Layer = Layer()
  public var outputLayer: Layer = Layer()
  
  public func generateRange() -> Float {
    return .random(in: minRange...maxRange)
  }
  
  public func setupInputLayer(neurons: Int, nextLayerNeuronsCount: Int) {
    for _ in 0..<neurons {
      let neuron = Neuron()
      for _ in 0..<nextLayerNeuronsCount {
        neuron.weights.append(generateRange())
      }
      inputLayer.neurons.append(neuron)
    }
  }
  
  public func setupHiddenLayer(neurons: Int) {
    for _ in 0..<neurons {
      let neuron = Neuron()
      neuron.weights.append(generateRange())
      hiddenLayer.neurons.append(neuron)
    }
  }
  
  public func setupOutputLayer() {
    outputLayer.neurons.append(Neuron())
  }
  
  public func predict(values: [Float]) -> [Float] {
    return forward(inputValues: values)
  }
  
  public func startTraining(data: [DataSet]) {
    for _ in 0..<ephos {
      for d in data {
        train(values: d.input, expected: d.expected)
      }
    }
    print("🟢 Training is over")
  }
  
  public func train(values: [Float], expected: Float) {
    let forward = forward(inputValues: values)
    guard let actual = forward.first else { return }
    let error: Float = actual - expected
    let sigmoidDx = actual * (1 - actual)
    let weightsDelta = error * sigmoidDx
    hiddenLayer.reverse(weightDelta: weightsDelta, learningRate: learningRate)
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
